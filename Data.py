from collections import defaultdict
import os
import re

from torch.utils.data import DataLoader, Subset

from torchvision.datasets import ImageFolder, CIFAR10, ImageNet
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms

from original_code.util.misc import get_rank, get_world_size

from Utils import *
import Misc

def data_path_to_data_name(data_path):
    if "imagenet" in data_path.lower():
        return "imagenet"

def data_str_to_num_classes(data_str):
    """Returns the number of classes in a dataset specified by [data_str].

    Note: this function requires hardcoding and won't work on arbitrary inputs.
    """
    if "imagenet" in data_str.lower():
        return 1000
    else:
        raise NotImplementedError()

def is_image_folder(f):
    """Returns if path [f] can be interpreted as an ImageFolder."""
    image_exts = [".png", ".jpg", ".jpeg"]
    def has_images(f):
        """Returns if [f] contains at least one image."""
        return (Misc.is_dir(f)
            and any([any([f_.lower().endswith(e) for e in image_exts])
                for f_ in os.listdir(f)]))
    return Misc.is_dir(f) and any([has_images(f"{f}/{d}") for d in os.listdir(f)])

def data_path_to_dataset(data_path, transform):
    """Returns an ImageFolder-like dataset over [data_path] with transform
    [transform].
    """
    if data_path.endswith(".lmdb"):
        return LMDBImageFolder(data_path, transform=transform)
    elif data_path.endswith(".tar"):
        tqdm.write(f"LOG: Constructing TAR dataset. This can take a bit...")
        return TarImageFolder(data_path,
            transform=transform,
            root_in_archive=os.path.splitext(os.path.basename(data_path))[0])
    elif is_image_folder(data_path):
        return ImageFolder(data_path, transform=transform)
    else:
        raise NotImplementedError(f"Could not match data_path {data_path}")


def data_path_to_loader(data_path, transform, distributed=False,
    shuffle=False, batch_size=1, pin_memory=False, num_workers=8, drop_last=False):
    """Returns a DataLoader or FFCV Loader over a dataset given by [data_path]
    with transform [transform].

    Args:
    data_path   -- path to an ImageFolder-compatible folder, or a .beton or
                    .lmdb file that can be interpreted as such
    transform   -- either a transforms.Compose() object, or, for .beton data
                    files, a data pipeline dictionary with 'image' and 'label'
                    keys
    distributed -- whether or not training is being done in a distributed fasion

    The remaining keyword arguments are identical to those for a DataLoader.
    """
    dataset = data_path_to_dataset(data_path, transform)
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle)
    elif not distributed and shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return DataLoader(dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last)