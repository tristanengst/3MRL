from collections import defaultdict
from copy import deepcopy
import re

from torch.utils.data import DataLoader, Subset

from torchvision.datasets import ImageFolder, CIFAR10, ImageNet
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms

from original_code.util.misc import get_rank, get_world_size

from Utils import *

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
        return (is_dir(f)
            and any([any([f_.lower().endswith(e) for e in image_exts])
                for f_ in os.listdir(f)]))
    return is_dir(f) and any([has_images(f"{f}/{d}") for d in os.listdir(f)])

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

class XYDataset(Dataset):
    """A simple dataset returning examples of the form (transform(x), y)."""

    def __init__(self, data, transform=None, target_transform=None, 
        normalize=False):
        """Args:
        data        -- a sequence of (x,y) pairs
        transform   -- the transform to apply to each returned x-value
        """
        super(XYDataset, self).__init__()
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        
        if self.normalize:
            means, stds = get_image_channel_means_stds(XYDataset(self.data))
            self.transform = transforms.Compose([transform,
                transforms.Normalize(means, stds, inplace=True)])

        if hasattr(self.data, "classes"):
            self.classes = deepcopy(self.data.classes)
        if hasattr(self.data, "class_to_idx"):
            self.classes = list(self.data.class_to_idx.keys())
            self.class_to_idx = deepcopy(self.data.class_to_idx)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = x if self.transform is None else self.transform(x)
        y = y if self.target_transform is None else self.target_transform(y)
        return x,y

def get_fewshot_dataset(dataset, n_way=5, n_shot=5, classes=None):
    """Returns a Subset of [dataset] giving a n-shot n-way task.

    Args:
    dataset -- ImageFolder-like dataset
    n_way   --
    n_shot
    classes --
    """
    if classes == "all":
        classes = set(dataset.targets)
    elif classes is None:
        classes = set(random.sample(dataset.classes, k=n_way))
    else:
        classes = set(classes)

    classes = {dataset.class_to_idx[c] for c in classes}
    class2idxs = defaultdict(lambda: [])
    for idx,t in enumerate(dataset.targets):
        if t in classes:
            class2idxs[t].append(idx)

    if not n_shot == "all":
        try:
            class2idxs = {c: random.sample(idxs, k=n_shot)
                for c,idxs in class2idxs.items()}
        except ValueError as e:
            class2n_idxs = "\n".join([f"\t{c}: {len(idxs)}"
                for c,idxs in class2idxs.items()])
            tqdm.write(f"Likely --val_n_shot asked for more examples than are available | val_n_shot {n_shot} | class to num idxs: {class2n_idxs}")
            raise e
  
    indices = flatten([idxs for idxs in class2idxs.values()])
    dataset = Subset(dataset, indices=indices)
    class2idx = {c: idx for idx,c in enumerate(sorted(classes))}
    return XYDataset(dataset, target_transform=lambda c: class2idx[c])
    

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
