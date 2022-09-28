# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit

from engine_finetune import train_one_epoch, evaluate

# We use FFCV as we can't put a plain ImageFolder directory on ComputeCanada
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, RandomHorizontalFlip, NormalizeImage, ModuleWrapper, Convert, Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder


def get_args_P():
    P = argparse.ArgumentParser("MAE linear probing for image classification", add_help=False)

    P.add_argument("--data_tr", required=True,
        help="Path to .beton file of training data")
    P.add_argument("--data_val", required=True,
        help="Path to .beton file of validation data")

    P.add_argument("--distributed", type=int, choices=[0, 1], default=1,
        help="Run with DDP or not")
    

    P.add_argument("--batch_size", default=512, type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    P.add_argument("--epochs", default=90, type=int,
        help="Number of epochs")
    P.add_argument("--accum_iter", default=1, type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")

    # Model parameters
    P.add_argument("--model", default="vit_large_patch16", type=str,
        metavar="MODEL",
        help="Name of model to train")

    # Optimizer parameters
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

    # * Finetuning params
    P.add_argument("--finetune", default="",
        help="finetune from checkpoint")
    P.add_argument("--global_pool", action="store_true")
    P.set_defaults(global_pool=False)
    P.add_argument("--cls_token", action="store_false", dest="global_pool",
        help="Use class token instead of global pool for classification")

    # Dataset parameters
    P.add_argument("--data_path", default="/datasets01/imagenet_full_size/061417/", type=str,
        help="dataset path")
    P.add_argument("--nb_classes", default=1000, type=int,
        help="number of the classification types")

    P.add_argument("--output_dir", default="./output_dir",
        help="path where to save, empty for no saving")
    P.add_argument("--log_dir", default="./output_dir",
        help="path where to tensorboard log")
    P.add_argument("--device", default="cuda",
        help="device to use for training / testing")
    P.add_argument("--seed", default=0, type=int)
    P.add_argument("--resume", default="",
        help="resume from checkpoint")

    P.add_argument("--start_epoch", default=0, type=int, metavar="N",
        help="start epoch")
    P.add_argument("--eval", action="store_true",
        help="Perform evaluation only")
    P.add_argument("--dist_eval", action="store_true", default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor")
    P.add_argument("--num_workers", default=10, type=int)
    P.add_argument("--pin_mem", action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    P.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    P.set_defaults(pin_mem=True)

    # distributed training parameters
    P.add_argument("--world_size", default=1, type=int,
        help="number of distributed processes")
    P.add_argument("--local_rank", default=-1, type=int)
    P.add_argument("--dist_on_itp", action="store_true")
    P.add_argument("--dist_url", default="env://",
        help="url used to set up distributed training")

    return P


def main(args):
    if args.distributed:
        misc.init_distributed_mode(args)
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # linear probe: weak augmentation


    ############################################################################
    # Set up FFCV DataLoaders
    ############################################################################
    imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
    imagenet_std = np.array([0.229, 0.224, 0.225]) * 255
    default_crop_ratio = 224/256

    class MakeChannelsLast(nn.Module):

        def __init__(self): super().__init__()

        def forward(self, x):
            print(x.shape)
            assert 0

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(device)]
    image_pipeline_tr = [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device),
        ToTorchImage(channels_last=False),
        Convert(torch.float32),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    image_pipeline_val = [
        CenterCropRGBImageDecoder((224, 224), ratio=default_crop_ratio),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device),
        ToTorchImage(channels_last=False),
        Convert(torch.float32),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
       ]

    data_loader_train = Loader(args.data_tr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=OrderOption.RANDOM,
        pipelines={"image": image_pipeline_tr, "label": label_pipeline},
        distributed=args.distributed,
        os_cache=True)
    
    data_loader_val = Loader(args.data_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=OrderOption.RANDOM,
        pipelines={"image": image_pipeline_val, "label": label_pipeline},
        distributed=args.distributed,
        os_cache=True)

    ############################################################################
    ############################################################################
    ############################################################################

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

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

    print("number of params (M): %.2f" % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
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
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

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
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_P()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
