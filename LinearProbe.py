import argparse
from itertools import product
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from Data import *
from utils.UtilsContrastive import *
from utils.Utils import *
from SimpleSaving import load_resnet_from_simple_save

# The following is needed for running happily on ComputeCanada
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

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

def get_fewshot_dataset(dataset, args):
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
        super(DensityDataset, self).__init__()
        self.args = args
        self.data = dataset
        self.feats = []

        loader = DataLoader(dataset,
            shuffle=False,
            batch_size=args.code_bs,
            pin_memory=True,
            num_workers=24)

        with torch.no_grad():
            for x,_ in tqdm(loader,
                desc="Building DensityDataset",
                dynamic_ncols=True,
                leave=False):

                z = sample_latent_dict(latent_spec, len(x) * args.codes_per_ex)
                self.features.append(model.forward_features(x, z).cpu())

        self.feats = torch.cat(self.feats, dim=0)

    def __len__(self): return len(self.dataset) * self.args.codes_per_ex

    def __getitem__(self, idx):
        return self.feats[idx], self.data[idx // self.args.codes_per_ex]

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
    loss_fn = nn.CrossEntropyLoss().to(device)
    accs_te = []
    losses_fn = []
    for trial in tqdm(range(args.val_trials),
        desc="Validation trials",
        dynamic_ncols=True):

        data_fn = OneAugDensityDataset(model, data_fn, latent_spec, args)
        data_te = OneAugDensityDataset(model, data_te, latent_spec, args)
        loader_fn = DataLoader(data_fn,
            batch_size=args.eval_bs,
            num_workers=24,
            pin_memory=True,
            shuffle=True)
        loader_te = DataLoader(data_te,
            batch_size=args.eval_bs,
            num_workers=24,
            pin_memory=True,
            shuffle=True)

        probe = nn.Linear(data_fn[0][0].shape[-1], num_classes)
        probe = nn.DataParallel(probe, device_ids=args.gpus).to(device)
        optimizer = AdamW(probe, lr=args.eval_lr, weight_decay=1e-6)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=args.epochs,
            max_lr=args.eval_lr,
            min_lr=args.eval_lr / 100)

        for epoch in tqdm(range(args.val_epochs),
            desc="Validation epochs",
            dynamic_ncols=True,
            leave=False):

            for idx,y in tqdm(loader_fn,
                desc="Validation batches",
                dynamic_ncols=True,
                leave=False):

                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(probe(x), y.to(device, non_blocking=True))
                loss.backward()
                optimizer.step()

            scheduler.step()

        losses_fn.append(loss.item())
        accs_te.append(accuracy(probe, loader_te))

    return {"accuracy/test": np.mean(accs_te),
        "a": np.std(accs_te) * 1.96 / np.sqrt(args.trials),
        "finetune_losses": np.mean(losses_fn),
        "finetune_losses_conf": np.std(losses_fn) * 1.96 / np.sqrt(args.trials)}

def parse_args():
    P = argparse.ArgumentParser()
    args = P.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
