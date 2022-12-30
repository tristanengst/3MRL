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
        return Misc.LMDBImageFolder(data_path, transform=transform)
    elif data_path.endswith(".tar"):
        tqdm.write(f"LOG: Constructing TAR dataset. This can take a bit...")
        return Misc.TarImageFolder(data_path,
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

def get_fewshot_dataset(dataset, n_way=5, n_shot=5, classes=None, seed=0,   
    fewer_shots_if_needed=False):
    """Returns a Subset of [dataset] giving a n-shot n-way task.

    Args:
    dataset                 -- ImageFolder-like dataset
    n_way                   -- number of classes to use
    n_shot                  -- number of shots to use
    classes                 -- classes to use (overrides [n_way])
    fewer_shots_if_needed   -- if [dataset] doesn't have all the [n_shots] for a
                                class, use less than [n_shots]
    """
    use_all_list = ["all", -1]
    
    if classes in use_all_list and n_shot in use_all_list:
        return dataset

    if classes in use_all_list:
        classes = set(dataset.classes)
    elif classes is None:
        n_way = len(dataset.classes) if n_way in use_all_list else n_way
        classes = set(Misc.sample(dataset.classes, k=n_way, seed=seed))
    else:
        classes = set(classes)

    classes = {dataset.class_to_idx[c] for c in classes}
    class2idxs = defaultdict(lambda: [])
    for idx,t in enumerate(dataset.targets):
        if t in classes:
            class2idxs[t].append(idx)

    if not n_shot in use_all_list:
        n_shot_fn = lambda x: (min(len(x), n_shot) if fewer_shots_if_needed else n_shot)
        try:
            class2idxs = {c: Misc.sample(idxs, k=n_shot_fn(idxs), seed=seed)
                for c,idxs in class2idxs.items()}
        except ValueError as e:
            class2n_idxs = "\n".join([f"\t{c}: {len(idxs)}"
                for c,idxs in class2idxs.items()])
            tqdm.write(f"Likely --val_n_shot asked for more examples than are available | val_n_shot {n_shot} | class to num idxs: {class2n_idxs}")
            raise e
  
    indices = Misc.flatten([idxs for idxs in class2idxs.values()])
    return ImageFolderSubset(dataset, indices=indices)

class ImageFolderSubset(Dataset):
    """Subset of an ImageFolder that preserves key attributes. Besides preserving ImageFolder-like attributes, the key improvement over a regular Subset is a target2idx dictionary that maps a target returned from [data] to a number in
    [0, len(classes)) which is necessary for classification.

    Doing this efficiently is oddly non-trivial.

    Besides maintaining [targets], [classes] and [class_to_idx] attributes,
    there are several key constraints:
    1) Every element of the [class_to_idx.values()] is a member of [targets]. As
        [targets] are integers in [0...N-1], this means that neither [targets]
        nor [class_to_idx] this attribute is not preserved by constructing an ImageFolderSubset
    2) Constructing this subset yields a dataset whose [classes] attribute is a
        subset of the same attribute of [data]

    With this constraint met, it's possible to construct this dataset on top of
    itself any number of times.

    Args:
    data    -- ImageFolder-like dataset
    indices -- list giving subset indices
    """

    def __init__(self, data, indices):
        super(ImageFolderSubset, self).__init__()
        self.data = data
        self.root = self.data.root
        self.indices = indices

        idxs_set = set(indices)

        # Mapping from indices we care about to the targets they have in [data]
        data_idx2target = {idx: t
            for idx,t in enumerate(data.targets)
            if idx in idxs_set}
        
        # Unique targets in subset of data
        data_targets = set(data_idx2target.values())

        # Mapping from unique targets in subset of data to their class
        data_target2class = {t: c for c,t in data.class_to_idx.items()
            if t in data_targets}

        # Mapping from indices we care about to their classes
        data_idx2class = {idx: data_target2class[t]
            for idx,t in enumerate(data.targets)
            if idx in idxs_set}

        # Subset of the classes in [data]
        self.classes = set(data_target2class.values())
        self.class_to_idx = {c: idx for idx,c in enumerate(sorted(self.classes))}
        self.data_target2idx = {t: idx for idx,t in enumerate(sorted(data_targets))}
        self.targets = [self.data_target2idx[t] for t in data_idx2target.values()]

    def __str__(self): return f"{self.__class__.__name__} [root={self.root} | length={self.__len__()} classes={self.classes}]"

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        x,y = self.data[self.indices[idx]]
        return x, self.data_target2idx[y]
