import argparse
import random
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from original_code.util.pos_embed import interpolate_pos_embed
import original_code

from Augmentation import *
from Data import *
from Models import *
from Utils import *

def accuracy(model, loader):
    """Returns the accuracy of [model] on data in [loader]."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            preds = torch.argmax(model(x.to(device)), dim=1)
            correct += torch.sum((preds == y.to(device))).item()
            total += len(preds)

    return correct / total

class FeatureDataset(Dataset):
    """Dataset of features for a fast linear probe. In contrast to a normal
    linear probe, an example's feature representation is memoized, meaning it is 
    seen with only one augmentation. This yields a lower accuracy than a normal 
    linear probe, but is orders of magnitude faster.

    Args:
    source          -- ImageFolder-like dataset with augmentations implemented
    model           -- NN whose forward method does feature extraction
    num_workers     -- number of workers to fetch data to build features for
    bs              -- batch size to get features in
    ignore_z        -- whether to ignore codes when building the dataset
    """
    def __init__(self, source, model, num_workers=24, bs=128, ignore_z=False):
        super(FeatureDataset, self).__init__()

        self.source = source
        loader = DataLoader(source,
            pin_memory=True,
            num_workers=num_workers,
            batch_size=bs)
        
        self.feats = []
        self.labels = []
        with torch.no_grad():
            for x,y in tqdm(loader,
                desc=f"BUILDING FEATURE DATASET | ignore_z [{bool(ignore_z)}]",
                leave=False,
                dynamic_ncols=True):

                x.to(device, non_blocking=True)

                # This assumes that [ignore_z] defaults to False, but allows
                # usage of models that don't support the kwarg
                fx = model(x, ignore_z=ignore_z) if ignore_z else model(x)
                
                self.feats.append(torch.squeeze(fx.cpu()))
                self.labels.append(y)
        
        self.feats = torch.cat(self.feats)
        self.labels = torch.cat(self.labels)
        
    def __len__(self): return len(self.source)

    def __getitem__(self, idx): return self.feats[idx], self.labels[idx]


def fast_linear_probe(model, data_tr, data_val, args, classes=None):
    """Returns a linear probe of [data_tr] and [data_val] using the encoder of
    [model] as a backbone. This can be done in a few minutes, but isn't as
    accurate as the LinearProbe.py script, which can take up to a day to run.

    Args:
    model       -- MaskedVAEViT model to use as a backbone
    data_tr     -- Dataset of training data
    data_val    -- Dataset of validation data
    args        -- argparse Namespace
    ignore_z    -- whether to run with latent codes ignored. For most backbones,
                    this is equivalent to them being MAE models
    """
    model = de_dataparallel(model).cpu()

    if isinstance(model, MaskedVAEViT):
        backbone = VariationalViTBackbone(encoder_kwargs=model.encoder_kwargs,
            idx2v_method=model.idx2v_method,
            global_pool=args.global_pool,
            noise=args.noise)
        interpolate_pos_embed(backbone, model.state_dict())
        _ = backbone.load_state_dict(model.state_dict(), strict=False)
        backbone = backbone.to(device)
        backbone.set_latent_spec(mask_ratio=0,
            test_input=torch.ones(4, 3, args.input_size, args. input_size,
            device=device))
    else:
        backbone = model

    backbone = nn.DataParallel(backbone.cpu(), device_ids=args.gpus).to(device)

    # Get the data
    if classes is None:
        classes = random.sample(data_tr.classes, k=args.val_n_way)
    data_tr = get_fewshot_dataset(data_tr,
        n_shot=args.val_n_shot,
        classes=classes,
        seed=args.seed)
    data_val = get_fewshot_dataset(data_val,
        n_shot="all",
        classes=classes,
        seed=args.seed)

    data_tr = FeatureDataset(data_tr,
        model=backbone,
        num_workers=args.num_workers,
        bs=args.probe_bs_val,
        ignore_z=args.probe_ignore_z)
    data_val = FeatureDataset(data_val,
        model=backbone,
        num_workers=args.num_workers,
        bs=args.probe_bs_val,
        ignore_z=args.probe_ignore_z)
    loader_tr = DataLoader(data_tr,
        shuffle=True,
        batch_size=args.probe_bs,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True)
    loader_val = DataLoader(data_val,
        batch_size=args.probe_bs_val,
        pin_memory=True,
        num_workers=args.num_workers)
    
    del backbone

    # Get the probe and optimization utilities
    probe = nn.Linear(data_tr[0][0].shape[0], len(classes))
    probe = nn.DataParallel(probe, device_ids=args.gpus).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(),
        lr=args.probe_lr,
        weight_decay=1e-6)
    loss_fn = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.probe_epochs * len(loader_tr),
        warmup_steps=max(1, args.probe_epochs // 10) * len(loader_tr),
        min_lr=1e-6)

    # Train the probe
    for e in tqdm(range(args.probe_epochs),
        desc="FAST LINEAR PROBE: Epochs",
        leave=True,
        dynamic_ncols=True):

        for batch_idx,(x,y) in tqdm(enumerate(loader_tr),
            desc="FAST LINEAR PROBE: Batches",
            total=len(loader_tr),
            leave=False,
            dynamic_ncols=True):

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                fx = probe(x)
                loss = loss_fn(fx, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            scheduler.step()

        if (args.probe_eval_iter > 0
            and ((e % args.probe_eval_iter == 0 and e > 0)
            or (e == 0 and args.probe_eval_iter == 1)
            or e == args.probe_epochs - 1)):
            tqdm.write(f"LOG: Fast Linear Probe: Epoch {e} | fast_linear_probe/lr {scheduler.get_lr()[0]:.5e} | fast_linear_probe/loss_tr {loss.item():.5f} | fast_linear_probe/acc_te {accuracy(probe, loader_val):.5f}") 
    
    model = model.to(device)
    return accuracy(probe, loader_val)

def get_args(args=None):
    P = argparse.ArgumentParser()
    P.add_argument("--model", type=str, required=True,
        help="Path to model to use as a backbone, or string specifying it")

    P.add_argument("--probe_ignore_z",  type=int, default=1, choices=[0, 1],
        help="Whether to ignore the code in all linear probing")

    P.add_argument("--data_tr", default="data/imagenet/train.tar", 
        type=argparse_file_type,
        help="String specifying training data")
    P.add_argument("--data_val", default="data/imagenet/val.tar", 
        type=argparse_file_type,
        help="String specifying finetuning data")
    P.add_argument("--input_size", default=224, type=int,
        help="Size of (cropped) images to feed to the model")

    P.add_argument("--val_n_way", type=int, default=10,
        help="Number of classes in probe/finetune data")
    P.add_argument("--val_n_shot", type=int, default=1000,
        help="Number of examples per class in probe/finetune data")

    P.add_argument("--global_pool", type=int, default=1, choices=[0, 1],
        help="Whether to global pool in linear probing")
    P.add_argument("--norm_pix_loss", type=int, default=1, choices=[0, 1],
        help="Whether to predict normalized pixels")
    P.add_argument("--probe_bs", type=int, default=128,
        help="Linear probe training batch size")
    P.add_argument("--probe_bs_val", type=int, default=256,
        help="Linear probe test/data gathering batch size")
    P.add_argument("--probe_lr", type=float, default=1e-3,
        help="Linear probe base learning rate")
    P.add_argument("--probe_epochs", type=int, default=100,
        help="Linear probe number of epochs")
    P.add_argument("--noise", default="gaussian", choices=["gaussian", "zeros"],
        help="Kind of noise to add inside the model")

    P.add_argument("--num_workers", type=int, default=20,
        help="Number of DataLoader workers")
    P.add_argument("--gpus", nargs="+", default=[0, 1], type=int,
        help="Device IDs of GPUs to use")

    P.add_argument("--probe_eval_iter", type=int, default=1,
        help="Number of epochs between validation")
    return P.parse_args() if args is None else P.parse_args(args)

if __name__ == "__main__":
    args = get_args()
    pretty_print_args(args)

    if os.path.exists(args.model) and "mae_checkpoints" in args.model:
        if "base" in args.model:
            model = original_code.models_vit.__dict__["vit_base_patch16"](
                num_classes=data_str_to_num_classes(args.data_tr),
                global_pool=args.global_pool)
        elif "large" in args.model:
            model = original_code.models_vit.__dict__["vit_large_patch16"](
                num_classes=data_str_to_num_classes(args.data_tr),
                global_pool=args.global_pool)
        else:
            raise NotImplementedError()

        model = VisionTransformerBackbone(args.global_pool, **model.kwargs)
        checkpoint = torch.load(args.model, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        interpolate_pos_embed(model, checkpoint_model)
        _ = model.load_state_dict(checkpoint_model, strict=False)
    elif os.path.exists(args.model):
        resume = torch.load(args.model)
        encoder_kwargs = resume["encoder_kwargs"]
        idx2v_method = resume["idx2v_method"]
        model = MaskedVAEViT(idx2v_method=idx2v_method, **encoder_kwargs)
        model.load_state_dict(resume["model"])
        model = model.to(device)
    elif args.model == "resnet18":
        arch = resnet18(weights="IMAGENET1K_V1", progress=True)
        model = nn.Sequential(*[l for n,l in arch.named_children()
                if not n in ["fc"]])
    else:
        raise NotImplementedError()

    data_tr = data_path_to_dataset(args.data_tr,
        transform=get_train_transforms(args))
    data_val = data_path_to_dataset(args.data_val,
        transform=get_train_transforms(args))

    acc = fast_linear_probe(model, data_tr, data_val, args)
    tqdm.write(f"Log: final accuracy {acc:.5f}")

    



