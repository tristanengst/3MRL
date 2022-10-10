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


from torch.distributed.elastic.multiprocessing.errors import record

from Augmentation import *
from Data import *
from Models import VariationalViT
from Utils import *

def get_linear_probe_folder(args, make_folder=True):
    if os.path.basename(args.finetune).startswith("mae"):
        model = os.path.splitext(os.path.basename(args.finetune))[0]
    else:
        model = args.finetune
        model = f"{os.path.basename(os.path.dirname(model))}/epoch_{os.path.basename(model).replace('.pt', '')}_linear_probe"

    data = data_path_to_data_name(args.data_tr)
    folder = f"{project_dir}/models/{model}/{data}-bs{args.batch_size}"

    if make_folder:
        conditional_safe_make_directory(folder)
        if not os.path.exists(f"{folder}/config.json"):
            with open(f"{folder}/config.json", "w+") as f:
                json.dump(vars(args), f)
    return folder

def get_args(args=None):
    """Returns a Namespace giving the parameters to run with."""
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
    P.add_argument("--noise", choices=["zeros", "gaussian"], default="zeros",
        help="Noise type. Only variational models can use non-zero noise.")

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
    P.add_argument("--data_tr", required=True, type=argparse_file_type,
        help="Path to training data")
    P.add_argument("--data_val", required=True, type=argparse_file_type,
        help="Path to testing (validation) data")
    P.add_argument("--input_size", type=int, default=224,
        help="Resolution at which to feed data to the model")

    P.add_argument("--output_dir", default=False,
        help="path where to save, empty for no saving")
    P.add_argument("--log_dir", default=None, # Unused because we just care about printed outputs
        help="path where to tensorboard log")
    P.add_argument("--device", default="cuda",
        help="device to use for training / testing")
    P.add_argument("--seed", default=0, type=int)

    P.add_argument("--dist_eval", type=int, default=1, choices=[0, 1],
        help="Enable distributed evaluation")
    P.add_argument("--num_workers", default=12, type=int,
        help="Number of workers")

    # distributed training parameters
    P.add_argument("--world_size", default=2, type=int,
        help="number of distributed processes")
    P.add_argument("--local_rank", default=-1, type=int)
    P.add_argument("--dist_url", default="env://",
        help="url used to set up distributed training")

    args = P.parse_args() if args is None else P.parse_args(args)
    return args

@record
def main(args):
    args.output_dir = get_linear_probe_folder(args)

    misc.init_distributed_mode(args)

    if misc.is_main_process():
        tqdm.write(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
        tqdm.write(f"Args:\n{args}")

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Get our DataLoaders.
    loader_tr = data_path_to_loader(args.data_tr,
        transform=get_train_transforms(args),
        distributed=args.distributed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        shuffle=True)
    loader_val = data_path_to_loader(args.data_val,
        transform=get_test_transforms(args),
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

    ############################################################################
    # Instantiate the model
    ############################################################################
    if "mae_checkpoints" in args.finetune:
        model = original_code.models_vit.__dict__[args.model](
            num_classes=data_str_to_num_classes(args.data_tr),
            global_pool=args.global_pool)
    elif os.path.exists(args.finetune):
        checkpoint = torch.load(args.finetune, map_location="cpu")
        model = VariationalViT(encoder_kwargs=checkpoint["encoder_kwargs"],
            idx2v_method=checkpoint["idx2v_method"],
            num_classes=data_str_to_num_classes(args.data_tr),
            global_pool=args.global_pool,
            noise=args.noise)
    else:
        raise NotImplementedError()

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        tqdm.write(f"Load pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                tqdm.write(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        trunc_normal_(model.head.weight, std=0.01)
        tqdm.write(f"{msg}")

        if args.global_pool:
            assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
        model.head)
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    # If the model is a VariationalViT, set its latent specification. We can
    # then pretend it doesn't exist in subsequent training, so it's fully
    # compatible with existing training code.
    if isinstance(model, VariationalViT):
        model.set_latent_spec(mask_ratio=0,
            test_input=torch.ones(4, 3, args.input_size, args. input_size,
                device=device))

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
        test_stats = evaluate(loader_val, model, device)
        tqdm.write(f"Accuracy of the network on test images: {test_stats['acc1']:.1f}%")
        exit(0)

    tqdm.write(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and isinstance(loader_tr, DataLoader):
            loader_tr.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, loader_tr,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and epoch == args.epochs - 1:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(loader_val, model, device)
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
