import re

from torch.utils.data import DataLoader, Subset

from torchvision.datasets import ImageFolder, CIFAR10, ImageNet
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms

from original_code.util.misc import get_rank, get_world_size

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, RandomHorizontalFlip, NormalizeImage, ModuleWrapper, Convert, Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder

from Utils import *

def data_str_to_num_classes(data_str):
    """
    """
    if "imagenet" in data_str.lower():
        return 1000
    else:
        raise NotImplementedError()

def is_image_folder(f):
    """Returns if path [f] can be interpreted as an ImageFolder."""
    def has_images(f):
        """Returns if [f] contains at least one image."""
        return (is_dir(f)
            and any([any([f_.lower().endswith(e) for e in image_exts])
                for f_ in os.listdir(f)]))
    return is_dir(f) and any([has_images(f"{f}/{d}") for d in os.listdir(f)])

def get_available_datasets(data_path=data_dir):
    """Returns all available datasets. Recursively looks through the folder
    hierarchy under [data_path], and returns all directories that can be
    interpreted as ImageFolders and all .beton files, .tar, and .lmdb files.

    Args:
    data_path   -- root folder to look inside
    """
    image_exts = [".png", ".jpeg", ".jpg"]
    image_collection_exts = [".beton", ".lmdb", ".tar"]

    def is_dir(f):
        """Returns if [f] is a directory."""
        try:
            _ = os.listdir(f)
            return True
        except NotADirectoryError as e:
            return False

    if is_image_folder(data_path):
       return [data_path]
    elif is_dir(data_path):
        return flatten([get_available_datasets(f"{data_path}/{d}")
            for d in os.listdir(data_path)])
    elif any([data_path.endswith(ext) for ext in image_collection_exts]):
        return [data_path]
    else:
        return []

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
    if data_path.endswith(".beton"):
        label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(get_rank())]
        order = OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL
        return Loader(data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            pipelines={"image": transform, "label": label_pipeline},
            distributed=distributed,
            os_cache=True)

    elif is_image_folder(data_path) or data_path.endswith(".lmdb"):
        if data_path.endswith(".lmdb"):
            dataset = LMDBImageFolder(data_path, transform=transform)
        else:
            dataset = ImageFolder(data_path, transform=transform)

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
    else:
        raise NotImplementedError()



class XYDataset(Dataset):
    """A simple dataset returning examples of the form (transform(x), y)."""

    def __init__(self, data, transform=transforms.ToTensor(), normalize=False):
        """Args:
        data        -- a sequence of (x,y) pairs
        transform   -- the transform to apply to each returned x-value
        """
        super(XYDataset, self).__init__()
        self.transform = transform
        self.data = data

        self.normalize = normalize
        if self.normalize:
            means, stds = get_image_channel_means_stds(XYDataset(self.data))
            self.transform = transforms.Compose([transform,
                transforms.Normalize(means, stds, inplace=True)])

        if hasattr(self.data, "classes"):
            self.classes = self.data.classes
        elif hasattr(self.data, "class_to_idx"):
            print(self.data.targets)
            self.classes = list(self.data.class_to_idx.keys())
            assert len(self.classes) == 1000, len(self.classes)
        else:
            raise NotImplementedError()

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image), label

def get_fewshot_dataset(dataset, args):
    """Returns a Subset of [dataset] giving a k-shot n-way task.

    Args:
    dataset -- ImageFolder-like dataset
    args    -- Namespace containing relevant parameters
    """
    raise NotImplementedError()

def get_image_channel_means_stds(dataset, bs=1024):
    """Returns an (mu, sigma) tuple where [mu] and [sigma] are tensors in which
    the ith element gives the respective ith mean and standard deviation of the
    ith channel of images in [dataset].

    Args:
    dataset -- dataset returning (x,y) pairs with [x] a CxHxW tensor
    bs      -- batch size to use in the computation
    """
    loader = DataLoader(dataset,
        num_workers=24,
        batch_size=bs,
        pin_memory=True)

    means, stds = torch.zeros(3, device=device), torch.zeros(3, device=device)
    for x,_ in tqdm(loader,
        desc="PREPARING DATA: Finding image channel stats for standardication",
        dynamic_ncols=True,
        leave=False):

        bs, c, _, _ = x.shape
        x = x.to(device, non_blocking=True)
        means += torch.mean(x, dim=[0, 2, 3]) * bs
        stds += torch.std(x, dim=[0, 2, 3]) * bs

    means = means.cpu() / len(dataset)
    stds = stds.cpu() / len(dataset)
    tqdm.write(f"LOG: Found images means {means.tolist()} and stds {stds.tolist()}")
    return means, stds
