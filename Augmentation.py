import torch
from torchvision import transforms

from original_code.util.misc import get_rank

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
default_crop_ratio = 224 / 256

def get_pretrain_transforms(args):
    pass

def get_train_transforms(data_str, input_size):
    """Returns transforms for pretraining."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size,
            scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])


def get_test_transforms(data_str, input_size):
    """Returns transforms for testing."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
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
    if isinstance(x, list):
        return [de_normalize(x_, mean=mean, std=std) for x_ in x]
    else:
        mean = torch.tensor(mean, device=x.device)
        std = torch.tensor(std, device=x.device)
        if isinstance(x, torch.Tensor) and len(x.shape) == 3:
            mean = mean.unsqueeze(-1).unsqueeze(-1)
            std = std.unsqueeze(-1).unsqueeze(-1)
        elif isinstance(x, torch.Tensor) and len(x.shape) == 4:
            mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif isinstance(x, torch.Tensor) and len(x.shape) == 5:
            return [de_normalize(x_, mean=mean, std=std) for x_ in x]
        else:
            raise NotImplementedError()

        mean = mean.expand(*x.shape)
        std = std.expand(*x.shape)
        return torch.multiply(x, std) + mean
