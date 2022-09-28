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
from Utils import *

# The following is needed for running happily on ComputeCanada
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def eval_model_on_loader(model, loader):
    """Returns an (accuracy, average loss) tuple of [model] run on data from
    [loader].
    """
    correct, total, total_loss = 0, 0, 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum").to(device)
    with torch.no_grad():
        for x,y in loader:
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            fx = model(x)
            total_loss += loss_fn(fx, y)
            correct += torch.sum((torch.argmax(fx, dim=1) == y)).item()
            total += x.shape[0]

    return correct / total, total_loss.item() / total

def get_fewshot_dataset(dataset, args):
    """Returns a Subset of [dataset] giving a k-shot n-way task.

    Args:
    dataset -- ImageFolder-like dataset
    args    -- Namespace containing relevant parameters
    """
    raise NotImplementedError()

class OneAugDensityDataset(Dataset):
    """Dataset for linear probing with a single augmentation but with many
    representations generated from a single example.

    Args:
    model       -- VariationalViT model
    dataset     -- ImageFolder-like dataset with transforms equipped
    latent_spec -- latent spec for adding noise to the model
    args        -- Namespace with relevant parameters
    """
    def __init__(self, model, dataset, latent_spec, args):
        super(OneAugDensityDataset, self).__init__()
        self.args = args

        if args.num_val_ex == -1:
            idxs = torch.arange(len(dataset))
        else:
            idxs = torch.linspace(0, len(dataset) - 1, args.num_val_ex)

        self.data = Subset(dataset, indices=[int(idx) for idx in idxs.tolist()])
        self.feats = []

        loader = DataLoader(self.data,
            shuffle=False,
            batch_size=args.code_bs,
            pin_memory=True,
            num_workers=args.num_workers)

        with torch.no_grad():
            for x,_ in tqdm(loader,
                desc="Building OneAugDensityDataset",
                dynamic_ncols=True,
                leave=False):

                with torch.cuda.amp.autocast():
                    z = sample_latent_dict(latent_spec, len(x) * args.z_per_ex_eval)
                    fx = model.module.forward_features(x.to(device), z)
                self.feats.append(fx.cpu())

        self.feats = torch.cat(self.feats, dim=0)

    def __len__(self): return len(self.data) * self.args.z_per_ex_eval

    def __getitem__(self, idx):
        return self.feats[idx], self.data[idx // self.args.z_per_ex_eval][1]

def linear_probe_one_aug(model, data_fn, data_te, latent_spec, args):
    """Returns a dictionary giving the results of linear probing [model] with a
    single augmentation for each example.

    Args:
    model       -- A VAEViT model
    data_fn     -- ImageFolder-like dataset equipping transforms containing
                    finetuning data
    data_te     -- ImageFolder-like dataset equipping transforms containing
                    testing data
    latent_spec -- latent specification for [model]
    args        -- Namespace with relevant parameters
    """
    num_classes = len(data_fn.classes)

    print("num_classes", num_classes)

    loss_fn = nn.CrossEntropyLoss().to(device)

    accs_te, losses_fn, losses_te = [], [], []
    for trial in tqdm(range(args.trials),
        desc="Validation trials",
        dynamic_ncols=True):

        trial_data_fn = OneAugDensityDataset(model, data_fn, latent_spec, args)
        trial_data_te = OneAugDensityDataset(model, data_te, latent_spec, args)
        loader_fn = DataLoader(trial_data_fn,
            batch_size=args.val_bs,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)
        loader_te = DataLoader(trial_data_te,
            batch_size=args.val_bs,
            num_workers=args.num_workers,
            pin_memory=True)

        probe = nn.Linear(trial_data_fn[0][0].shape[-1], num_classes)
        probe = nn.DataParallel(probe, device_ids=args.gpus).to(device)
        optimizer = AdamW(probe.parameters(), lr=args.val_lr, weight_decay=1e-6)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=args.epochs,
            max_lr=args.val_lr,
            min_lr=args.val_lr / 100)

        for epoch in tqdm(range(args.val_epochs),
            desc="Validation epochs",
            dynamic_ncols=True,
            leave=False):

            for x,y in tqdm(loader_fn,
                desc="Validation batches",
                dynamic_ncols=True,
                leave=False):

                fx, y = probe(x), y.to(device, non_blocking=True)
                loss = loss_fn(fx, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                tqdm.write(f"            Accuracy {torch.sum((torch.argmax(fx, dim=1) == y)).item() / x.shape[0]}")

            scheduler.step()
            acc_te, loss_te = eval_model_on_loader(probe, loader_te)

            tqdm.write(f"acc_te {acc_te}")

        acc_te, loss_te = eval_model_on_loader(probe, loader_te)
        losses_fn.append(loss.item())
        losses_te.append(loss_te)
        accs_te.append(acc_te)

    return {"accuracy/lin_probe_test": np.mean(accs_te),
        "accuracy_lin_probe_test_conf": np.std(accs_te) * 1.96 / np.sqrt(args.trials),
        "loss/lin_probe_train": np.mean(losses_fn),
        "loss/lin_probe_test": np.mean(losses_te),
        "loss_lin_probe_test_conf": np.std(losses_fn) * 1.96 / np.sqrt(args.trials),
        "loss_lin_probe_test_conf": np.std(losses_te) * 1.96 / np.sqrt(args.trials)}

def get_args(args=None):
    P = argparse.ArgumentParser()
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="Type of W&B logging")
    P.add_argument("--resume", default=None,
        help="Path to checkpoint to resume")
    P.add_argument("--suffix", type=str, default=None,
        help="Optional suffix")

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

if __name__ == "__main__":
    args = parse_args()

    resume = torch.load(args.resume)
    model = resume["model"].to(device)
    model_args = resume["args"]
