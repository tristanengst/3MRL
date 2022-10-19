import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm as tqdm

from original_code.util.pos_embed import interpolate_pos_embed

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
    """
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
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for x,y in tqdm(loader,
                    desc=f"BUILDING FEATURE DATASET | ignore_z [{bool(ignore_z)}]",
                    leave=False,
                    dynamic_ncols=True):

                    fx = model(x.to(device, non_blocking=True), ignore_z=ignore_z)
                    self.feats.append(fx.cpu())
                    self.labels.append(y)
        
        self.feats = torch.cat(self.feats)
        self.labels = torch.cat(self.labels)
        
    def __len__(self): return len(self.source)

    def __getitem__(self, idx): return self.feats[idx], self.labels[idx]


def fast_linear_probe(model, data_tr, data_val, args, ignore_z=False):
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

    # Initialize the backbone architecture
    backbone = VariationalViTBackbone(encoder_kwargs=model.encoder_kwargs,
        idx2v_method=model.idx2v_method,
        global_pool=args.global_pool,
        noise=args.noise)
    interpolate_pos_embed(backbone, model.state_dict())
    _ = backbone.load_state_dict(model.state_dict(), strict=False)
    bakcbone = backbone.to(device)
    backbone.set_latent_spec(mask_ratio=0,
        test_input=torch.ones(4, 3, args.input_size, args. input_size,
        device=device))
    backbone = nn.DataParallel(backbone.cpu(), device_ids=args.gpus).to(device)

    # Get the data
    num_classes = len(data_tr.classes)
    data_tr = Subset(data_tr, indices=list(range(args.probe_ex_tr)))
    data_tr = FeatureDataset(data_tr,
        model=backbone,
        num_workers=args.num_workers,
        bs=args.probe_bs_val,
        ignore_z=ignore_z)
    data_val = Subset(data_val, indices=list(range(args.probe_ex_val)))
    data_val = FeatureDataset(data_val,
        model=backbone,
        num_workers=args.num_workers,
        bs=args.probe_bs_val,
        ignore_z=ignore_z)
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
    probe = nn.Linear(data_tr[0][0].shape[0], num_classes)
    probe = nn.DataParallel(probe, device_ids=args.gpus).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(),
        lr=args.probe_lr,
        weight_decay=1e-6)
    loss_fn = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

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
    
    model = model.to(device)
    return accuracy(probe, loader_val)


