"""Heavily modified script for linear probing. The key changes are
(1) This script can work with VariationalViT models, thanks to their ability to
    implement the same API for forward() as ViT models
(2) Data can be loaded from different kinds of datasets

I do not endorse the code quality present.
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
from timm.models.layers import trunc_normal_

import original_code.util.misc as misc
from original_code.util.pos_embed import interpolate_pos_embed
from original_code.util.misc import NativeScalerWithGradNormCount as NativeScaler
from original_code.util.lars import LARS
from original_code.util.crop import RandomResizedCrop
import original_code.models_vit

from original_code.engine_finetune import train_one_epoch, evaluate

from Augmentation import *
from Data import *
from Models import VariationalViT
from Utils import *

def get_args(args=None):
    """Returns a Namespace giving the parameters to run with."""
    file_if_exists = lambda f: f if os.path.exists(f) else False

    P = argparse.ArgumentParser()
    P.add_argument("--eval", action="store_true",
        help="Perform evaluation only")

    # Model parameters
    P.add_argument("--model", default="vit_base_patch16", type=str,
        help="Name of backbone")
    P.add_argument("--finetune", default="", required=True,
        help="checkpoint to finetune from")
    P.add_argument("--global_pool", type=int, default=0, choices=[0, 1],
        help="Whether to use global pooling in forming representations")
    P.add_argument("--resume", default="",
        help="...")

    # Training parameters
    P.add_argument("--batch_size", default=512, type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    P.add_argument("--epochs", default=90, type=int)
    P.add_argument("--accum_iter", default=1, type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    P.add_argument("--weight_decay", type=float, default=0,
        help="weight decay (default: 0 for linear probe following MoCo v1)")
    P.add_argument("--lr", type=float, default=None, metavar="LR",
        help="learning rate (absolute lr)")
    P.add_argument("--blr", type=float, default=0.1, metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    P.add_argument("--min_lr", type=float, default=0., metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0")
    P.add_argument("--warmup_epochs", type=int, default=10, metavar="N",
        help="epochs to warmup LR")
    P.add_argument("--start_epoch", default=0, type=int, metavar="N",
        help="start epoch")

    # Dataset parameters
    P.add_argument("--data_tr", required=True, type=file_if_exists,
        help="Path to training data")
    P.add_argument("--data_val", required=True, type=file_if_exists,
        help="Path to testing (validation) data")
    P.add_argument("--input_size", type=int, default=224,
        help="Resolution at which to feed data to the model")

    P.add_argument("--output_dir", default="./output_dir",
        help="path where to save, empty for no saving")
    P.add_argument("--log_dir", default=None,
        help="path where to tensorboard log")
    P.add_argument("--device", default="cuda",
        help="device to use for training / testing")
    P.add_argument("--seed", default=0, type=int)

    P.add_argument("--dist_eval", type=int, default=1, choices=[0, 1],
        help="Enable distributed evaluation")
    P.add_argument("--num_workers", default=24, type=int,
        help="Number of workers")

    # distributed training parameters
    P.add_argument("--world_size", default=2, type=int,
        help="number of distributed processes")
    P.add_argument("--local_rank", default=-1, type=int)
    P.add_argument("--dist_url", default="env://",
        help="url used to set up distributed training")

    args = P.parse_args() if args is None else P.parse_args(args)
    return args


def main(args):
    misc.init_distributed_mode(args)

    tqdm.write(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    tqdm.write(f"Args:\n{args}")

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Get our DataLoaders. This is done in a way that's agnostic to the format
    # the data is stored in.
    data_loader_train = data_path_to_loader(args.data_tr,
        transform=get_train_transforms(args.data_tr, args.input_size),
        distributed=args.distributed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        shuffle=True)
    data_loader_val = data_path_to_loader(args.data_val,
        transform=get_test_transforms(args.data_val, args.input_size),
        distributed=args.distributed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        shuffle=True)

    if misc.get_rank() == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    model = original_code.models_vit.__dict__[args.model](
        num_classes=data_str_to_num_classes(args.data_tr),
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        tqdm.write(f"Load pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                tqdm.write(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        tqdm.write(f"{msg}")

        if args.global_pool:
            assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model"s head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # tqdm.write("Model = %s" % str(model_without_ddp))
    tqdm.write("number of params (M): %.2f" % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    tqdm.write("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    tqdm.write("actual lr: %.2e" % args.lr)

    tqdm.write("accumulate grad iterations: %d" % args.accum_iter)
    tqdm.write("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tqdm.write(f"{optimizer}")
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    tqdm.write("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        tqdm.write(f"Accuracy of the network on test images: {test_stats['acc1']:.1f}%")
        exit(0)

    tqdm.write(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and isinstance(data_loader_train, DataLoader):
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        tqdm.write(f"Accuracy of the network on test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        tqdm.write(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"test_{k}": v for k, v in test_stats.items()},
                        "epoch": epoch,
                        "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    tqdm.write("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
