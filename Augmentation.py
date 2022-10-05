import torch
from torchvision import transforms

from original_code.util.misc import get_rank

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
default_crop_ratio = 224 / 256

def get_pretrain_transforms(args):
    pass

def get_train_transforms(args):
    """Returns transforms for pretraining."""
    return transforms.Compose([
        transforms.RandomResizedCrop(args.input_size,
            scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])


def get_test_transforms(args):
    """Returns transforms for testing."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])


def de_normalize(x, mean=imagenet_mean, std=imagenet_std):
    """Returns [x] with the images therein denormalized with [mean] and [std].

    Args:
    x       -- list of tensors or tensor. The last three dimensions of each
                tensor in [x] or of [x] are taken to contain an image
    mean    -- list giving the per-channel mean
    std     -- list giving the per-channel standard deviation
    """
    mean = [-1 * m for m in mean]
    std = [1/s for s in std]
    return transforms.functional.normalize(
        transforms.functional.normalize(x,
            mean=[0, 0, 0],
            std=std),
        mean=mean,
        std=[1, 1, 1])
