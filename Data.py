import re

from torch.utils.data import DataLoader, Subset


from torchvision.datasets import ImageFolder, CIFAR10
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms

from Utils import *

# Use the augmentation policy of the original MAE paper's linear probing.
def get_train_transforms(args):
    return transforms.Compose([
            transforms.RandomResizedCrop(args.input_size,
                scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

def get_finetuning_transforms(args):
    return transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

def get_test_transforms(args):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

def get_available_datasets(data_dir=f"./data"):
    """Returns a list of all available datasets that can be interpreted by
    get_imagefolder_data().
    """
    def without_res(d):
        """Returns dataset string [d] without the resolution specified."""
        matches = re.findall("_[0-9]+x[0-9]+", d)
        if len(matches) > 1:
            raise ValueError()
        elif len(matches) == 1:
            return d.replace(matches[0], "")
        else:
            return d.replace()

    dirs = [f"{data_dir}/{f}" for f in os.listdir(data_dir)]
    image_folder_dirs = [d for d in dirs if is_image_folder(d)]
    lmdb_dirs = [f"{d}/{f}" for d in dirs for f in os.listdir(d)
        if os.path.isdir(d) and "lmdb" in d and f.endswith(".lmdb")]
    options = image_folder_dirs + lmdb_dirs
    options = [o.replace(f"{data_dir}/", "") for o in options]
    options = [o.replace(".lmdb", "") for o in options]
    options = [without_res(o) for o in options]
    return options

def get_imagefolder_data(*datasets, res=256, data_path=data_dir):
    """Returns a transform-free dataset for each path in [*datasets].

    Args:
    datasets            -- list of strings that can be interpreted as a dataset,
                            or a path to a folder that can be interpreted as a
                            PreAugmentedDataset
    res                 -- list of resolutions or a list resolution interpreted
                            as a single-item list
    data_path           -- path to where datasets are found

    Returns:
    Roughly, [[d(p, r) for r in res] for p in datasets], where [d(.,.)] maps
    a string identifying a folder of data and a resolution to a PyTorch dataset
    over the data. If [res] contains only one item, the sublist is flattened.
    """
    ignored_data_strs = ["cifar10/train", "cifar10/test", "cv"]

    def contains_augs(data_str):
        """Returns if any images in [data_str] are augmentations."""
        if "lmdb" in data_str:
            return lmdb_file_contains_augs(data_str)
        else:
            return any(["_aug" in image for label in os.listdir(data_str)
                for image in os.listdir(f"{data_str}/{label}")])

    def data_str_with_resolution(data_str, res):
        """Returns [data_str] at resolution [res].

        Args:
        data_str -- a path to something that could be turned into an ImageFolder
        res     -- the desired resolution
        """
        if (data_str in ignored_data_strs
            or has_resolution(data_str)
            or os.path.exists(data_str)):
            return data_str
        elif "lmdb" in data_str:
            dataset = data_str.strip("/")
            idx = data_str.rindex("/")
            return data_str[:idx] + f"_{res}x{res}" + data_str[idx:] + ".lmdb"
        else:
            dataset = data_str.strip("/")
            idx = data_str.rindex("/")
            return data_str[:idx] + f"_{res}x{res}" + data_str[idx:]

    def data_str_to_dataset(data_str):
        """Returns the dataset that can be built with [data_str].

        Args:
        data_str    -- One of 'cv', 'cifar10/train', 'cifar10/test', or a path
                        to a folder over which an ImageFolder can be constructed
        """
        if data_str == "cifar10/train":
            return CIFAR10(root=data_path, train=True, download=True)
        elif data_str == "cifar10/test":
            return CIFAR10(root=data_path, train=False, download=True)
        elif data_str == "cv":
            return "cv"
        else:
            if os.path.exists(data_str):
                data_str = data_str
            elif os.path.exists(f"{data_path}/{data_str}"):
                data_str = f"{data_path}/{data_str}"
            else:
                raise ValueError(f"Couldn't find a folder for data string {data_str}")

            if contains_augs(data_str):
                # PreAugmentedDatasets can handle LMDB files as data sources
                return PreAugmentedDataset(data_str, verbose=False)
            elif "lmdb" in data_str:
                return LMDBImageFolder(data_str)
            else:
                return ImageFolder(data_str)

    res = [res] if isinstance(res, int) else res
    datasets = [[data_str_with_resolution(d, r) for r in res] for d in datasets]
    datasets = [[data_str_to_dataset(d) for d in d_] for d_ in datasets]
    result = tuple([d[0] if len(d) == 1 else d for d in datasets])
    return result[0] if len(result) == 1 else result

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

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image), label

class FeatureDataset(Dataset):
    """A dataset of model features.

    Args:
    F       -- a feature extractor, eg. the backbone of a ResNet
    data    -- a dataset of XY pairs
    bs      -- the batch size to use for feature extraction
    """

    def __init__(self, data, F, bs=1000, num_workers=24):
        super(FeatureDataset, self).__init__()
        loader = DataLoader(data,
            batch_size=bs,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers)

        data_x, data_y = [], []
        F = F.to(device)
        F.eval()
        with torch.no_grad():
            for x,y in tqdm(loader,
                desc="Building FeatureDataset",
                leave=False):
                data_x.append(F(x.to(device)).cpu())
                data_y.append(y)

        data_x = [x for x_batch in data_x for x in x_batch]
        data_y = [y for y_batch in data_y for y in y_batch]
        self.data = list(zip(data_x, data_y))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]

class ImageCodeDataset(Dataset):
    """Dataset that returns a (cx, codes, y) tuple where [cx] is a CxHxW
    corrupted image, [codes] is a list of codes for generating decorruptions of
    the image at progressively higher resolutions, and [y] is a list of the
    original image a progressively higher resolutions.

    To iterate over this dataset more than once, construct a DataLoader over it,
    and use the itertools.chain() method on a list of this DataLoader,
    duplicated a bunch.

    Args:
    cx      -- BSxCxHxW tensor of corrupted images
    codes   -- list of codes of shape BSxCODE_DIM. Elements in the list should
                be codes for sequentially greater resolutions
    ys      -- list of BSxCxHxW target images. Elements in the list should be
                for sequentially greater resolutions
    """
    def __init__(self, cx, codes, ys):
        super(ImageCodeDataset, self).__init__()
        assert len(codes) == len(ys)
        assert all([len(c) == len(y) == cx.shape[0] for c,y in zip(codes, ys)])

        self.cx = cx.cpu()
        self.codes = [c.cpu() for c in codes]
        self.ys = [y.cpu() for y in ys]

    def __len__(self): return len(self.cx)

    def __getitem__(self, idx):
        cx = self.cx[idx]
        codes = [c[idx] for c in self.codes]
        ys = [y[idx] for y in self.ys]
        return cx, codes, ys

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
