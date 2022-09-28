import argparse
from itertools import product
import os
from tqdm import tqdm

import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, Dataset
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.cuda.amp import GradScaler, autocast

from Augmentation import *
from FFCVData import get_ffcv_loader
from Models import *
from Utils import *

from timm.models.layers import trunc_normal_
from original_code.util.lars import LARS
from original_code.util.misc import NativeScalerWithGradNormCount as NativeScaler

# The following is needed for running happily on ComputeCanada
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def eval_model_on_loader(model, loader, latent_spec, args):
    """Returns an (accuracy, average loss) tuple of [model] run on data from
    [loader].
    """
    correct, total, total_loss = 0, 0, 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum").to(device)
    with torch.no_grad():
        for x,y in tqdm(loader,
            desc="Validation: test results",
            leave=False,
            dynamic_ncols=True):

            try:
                with torch.cuda.amp.autocast():
                    y = y.to(device, non_blocking=True)
                    x = x.to(device, non_blocking=True)
                    z = sample_latent_dict(latent_spec, args.z_per_ex_eval, noise=args.noise)
                    fx = model(x, z)
                    total_loss += loss_fn(fx, y)
                    correct += torch.sum((torch.argmax(fx, dim=1) == y)).item()
                    total += x.shape[0]
            except:
                print(torch.argmax(fx, dim=1))
                print(y)

    return correct / total, total_loss.item() / total

def linear_probe(args):
    """Returns a dictionary giving the results of linear probing [model] with a
    single augmentation for each example.

    Args:
    model       -- A MaskedVAEViT model
    data_fn     -- ImageFolder-like dataset equipping transforms containing
                    finetuning data
    data_te     -- ImageFolder-like dataset equipping transforms containing
                    testing data
    latent_spec -- latent specification for [model]
    args        -- Namespace with relevant parameters
    """
    num_classes = data_str_to_num_classes(args.data_fn)
    loss_fn = nn.CrossEntropyLoss().to(device)
    loader_fn = get_ffcv_loader(args.data_tr, args, bs=args.val_bs)
    loader_te = get_ffcv_loader(args.data_fn, args, bs=args.val_bs)
    
    ############################################################################
    # Instantiate the model similarly to main_linprobe.py. The key difference is
    # that it's a VariationalViT.
    ############################################################################
    if args.variational:
        model = VariationalViT(parse_variational_spec(args),
            v_mae_model=load_model(args),
            num_classes=num_classes,
            global_pool=args.global_pool)
    else:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            global_pool=args.global_pool)
            

    model.head = torch.nn.Sequential(
        nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
        model.head)
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    model = nn.DataParallel(model, device_ids=args.gpus).to(device)

    latent_spec = model.module.get_latent_spec(
        torch.ones(4, 3, args.input_size, args.input_size, device=device))
    optimizer = AdamW(model.module.head.parameters(),
        lr=args.val_lr,
        weight_decay=0)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.epochs * len(loader_fn),
        warmup_steps=10 * len(loader_fn),
        max_lr=args.val_lr,
        min_lr=args.val_lr / 100)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(args.val_epochs),
        desc="Validation epochs",
        dynamic_ncols=True,
        leave=False):

        for x,y in tqdm(loader_fn,
            desc=f"Validation batches | lr {scheduler.get_lr()[0]:.5e}",
            dynamic_ncols=True,
            leave=False):

            with torch.cuda.amp.autocast():
                y = y.to(device, non_blocking=True)
                fx = model(x, noise=args.noise, z_per_ex=args.z_per_ex_eval)
                loss = loss_fn(fx, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            scheduler.step()

        acc_te, loss_te = eval_model_on_loader(model, loader_te, latent_spec, args)
        tqdm.write(f"    Linear probe epoch {epoch}/{args.val_epochs}: accuracy/test {acc_te:.3f} | loss/test {loss_te:.3f} | lr {scheduler.get_lr()[0]:.5e}")

    return {"accuracy/lin_probe_test": acc_te,
        "loss/lin_probe_train": losses_fn,
        "loss/lin_probe_test":losses_te,

def get_args(args=None):
    P = argparse.ArgumentParser()
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="Type of W&B logging")
    P.add_argument("--resume", default=None,
        help="Path to checkpoint to resume")
    P.add_argument("--suffix", type=str, default=None,
        help="Optional suffix")
    P.add_argument("--variational", type=int, default=1, choices=[0, 1],
        help="Whether to use a variational model or not")
    P.add_argument("--backbone", default="vit_base_patch16",
        help="Name of model backbone to use")

    # Data arguments
    P.add_argument("--data_tr", required=True, choices=get_available_datasets(),
        help="String specifying training data")
    P.add_argument("--data_fn", required=True, choices=get_available_datasets() + ["cv"],
        help="String specifying finetuning data")
    P.add_argument("--data_te", required=True, choices=get_available_datasets() + ["cv"],
        help="String specifying testing data")
    P.add_argument("--data_path", default=data_dir,
        help="Path to where datasets are stored")

    # Evaluation arguments. Optimization here isn't too important because we can
    # use saved models to get representations and do hyperparameter tuning.
    P.add_argument("--num_ex_for_eval_tr", default=8,
        help="Number of training examples for logging")
    P.add_argument("--num_ex_for_eval_te", default=8,
        help="Number of training examples for logging")
    P.add_argument("--z_per_ex", default=6,
        help="Number of latents per example for logging")
    P.add_argument("--z_per_ex_eval", type=int, default=16,
        help="Number of codes to use per example in linear probing")
    P.add_argument("--trials", type=int, default=1,
        help="Number of trials to use in linear probe")
    P.add_argument("--num_val_ex", type=int, default=1024,
        help="Number of examples to use for validation")
    P.add_argument("--val_epochs", type=int, default=120,
        help="Number of epochs to use in linear probing")
    P.add_argument("--val_bs", type=int, default=8,
        help="Batch size to use in supervised linear probing")
    P.add_argument("--val_lr", type=float, default=1e-3,
        help="Learning rate to use in linear probing")
    P.add_argument("--noise", default="gaussian", choices=["gaussian", "zeros"],
        help="Kind of noise to add inside the model")

    # Hardware arguments
    P.add_argument("--gpus", nargs="+", default=[0, 1], type=int,
        help="Device IDs of GPUs to use")
    P.add_argument("--job_id", type=int, default=None,
        help="SLURM job_id if available")
    P.add_argument("--num_workers", type=int, default=24,
        help="Number of DataLoader workers")

    args = P.parse_args() if args is None else P.parse_args(args)

    args.uid = wandb.util.generate_id() if args.job_id is None else args.job_id
    return args

def load_model(args):
    """Returns the model specified by [args]."""
    if args.variational:
        model = VariationalViT(parse_variational_spec(args),
            v_mae_model=load_model(args),
            num_classes=data_str_to_num_classes(args.data_fn),
            global_pool=args.global_pool)
    else:
        model = models_vit.__dict__[args.model](
            num_classes=data_str_to_num_classes(args.data_fn),
            global_pool=args.global_pool)

    checkpoint = torch.load(args.resume, map_location="cpu")
        checkpoint = checkpoint["model"]
        
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if (k in checkpoint and not checkpoint[k].shape == state_dict[k].shape):
            tqdm.write(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
    interpolate_pos_embed(model, checkpoint)

    # load pre-trained model
    message = model.load_state_dict(checkpoint_model, strict=False)
    tqdm.write(f"MODEL LOADING MESSAGE:\n{message}")

    if args.global_pool:
        assert set(message.missing_keys) == {"head.weight", "head.bias",
            "fc_norm.weight", "fc_norm.bias"}
    else:
        assert set(message.missing_keys) == {"head.weight", "head.bias"}

    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head.weight, std=0.01)

    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
        model.head)
    
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    
    return model.to(device)

    

if __name__ == "__main__":
    args = get_args()

    
    

