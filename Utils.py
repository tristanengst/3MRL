import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import io
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import torch
import torch.nn as nn
from torchvision.transforms import functional as tv_functional
import wandb

import Misc

from tqdm import tqdm

import Utils

# I don't think this does anything because we don't have convolutions, but it
# probably can't hurt
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["savefig.bbox"] = "tight"

def argparse_file_type(f):
    """Returns [f] if the file exists, else raises an ArgumentType error."""
    if os.path.exists(f):
        return f
    elif f.startswith("$SLURM_TMPDIR"):
        return f
    else:
        raise argparse.ArgumentTypeError(f"Could not find data file {f}")

def de_dataparallel(net):
    """Returns a reference to the network wrapped in a DataParallel object
    [net], or [net] if it isn't data parallel.
    """
    return net.module if isinstance(net, nn.DataParallel) else net

def images_to_pil_image(images):
    """Returns tensor datastructure [images] as a PIL image."""
    images = Misc.make_2d_list_of_tensor(images)

    fig, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images),
        squeeze=False)

    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            image = torch.clip((image * 255), 0, 255).int()
            image = torch.einsum("chw->hwc", image.detach().cpu())
            axs[i, j].imshow(np.asarray(image), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=256)
    buf.seek(0)
    plt.close("all")
    return Image.open(buf)

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    if isinstance(xs, list) or isinstance(xs, set) or isinstance(xs, tuple):
        result = []
        for x in xs:
            result += flatten(x)
        return result
    else:
        return [xs]

def scheduler_to_lrs(s):
    """Returns a mapping from parameter group names to their current learning
    rate for scheduler [s]. If the parameter group is unnamed, 
    """
    groups = [g["name"] if "name" in g else f"group_{idx}"
            for idx,g in enumerate(s.optimizer.param_groups)]
    lrs = [g["lr"] for g in s.optimizer.param_groups]
    return OrderedDict({g: lr for g,lr in zip(groups, lrs)})

class KOrKMinusOne:
    """Class for maintaining a condition on data [idxs] in which for natural
    number [k], each element has been returned from the pop() method either [k]
    or [k-1] times, regardless of the number of calls to pop().

    WARNING: This class is not thread-safe.

    Args:
    idxs    -- list of data points to return, meant to just be numbers
    shuffle -- whether or not to shuffle the order in which elements of [idx]
                are returned, while maintaining the condition
    """
    def __init__(self, idxs, shuffle=True, seed=0):   
        self.shuffle = shuffle
        self.counter = 0
        self.num_resets = 0
        self.seed = seed

        if shuffle:
            self.idxs = sample(idxs, k=len(idxs), seed=seed)
        else:
            self.idxs = idxs

    def pop(self):
        if self.counter == len(self.idxs):
            self.counter = 0
            self.num_resets += 1
            if self.shuffle:
                self.idxs = sample(self.idxs,
                    k=len(self.idxs),
                    seed=self.seed + self.num_resets)
            else:
                self.idxs = self.idxs

        result = self.idxs[self.counter]
        self.counter += 1
        return result

    def pop_k(self, k): return [self.pop() for _ in range(k)]

    def state_dict(self): return self.__dict__

    def __str__(self): return f"KOrKMinusOne [shuffle {self.shuffle} | counter {self.counter} | num_resets {self.num_resets} | seed {self.seed} | num_idxs {len(self.idxs)}]"

    @staticmethod
    def from_state_dict(state_dict):
        kkm = KOrKMinusOne([])
        kkm.__dict__.update(state_dict)
        return kkm

def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tqdm.write(f"Set the NumPy, PyTorch, and Random modules seeds to {seed}")
    elif isinstance(seed, dict):
        random.setstate(seed["random_seed"])
        np.random.set_state(seed["numpy_seed"])
        torch.set_rng_state(seed["torch_seed"])
        torch.cuda.set_rng_state(seed["torch_cuda_seed"])
        tqdm.write(f"Reseeded program with old seed")
    else:
        raise ValueError(f"Seed should be int or contain resuming keys")
    return seed

def sorted_namespace(args):
    """Returns argparse Namespace [args] after sorting the args in it by key
    value. The utility of this is printing.
    """
    d = vars(args)
    return argparse.Namespace(**{k: d[k] for k in sorted(d.keys())})

def conditional_make_folder(f):
    try:
        os.makedirs(f)
    except:
        pass

def sample(select_from, k=-1, seed=0):
    """Returns [k] items sampled without replacement from [select_from] with
    seed [seed], without changing the internal seed of the program. This
    explicitly ensures reproducability.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        result = random.sample(select_from, k=k)
    except ValueError as e:
        tqdm.write(f"Tried to sample {k} from {len(select_from)} things")
        raise e
    random.setstate(state)
    return result

def split_by_param_names(model, *param_names):
    name2params = {p: [] for p in param_names} | {"default": []}
    for k,v in model.named_parameters():
        found_custom_name = False
        for p in param_names:
            if p in k:
                name2params[p].append(v)
                found_custom_name = True
                break
        if not found_custom_name:
            name2params["default"].append(v)

    return [{"params": p, "name": n} for n,p in name2params.items()]

class StepScheduler:
    """StepLR but with easier control.
    
    Args:
    optimizer   -- optimizer to step
    lrs         -- list where sequential pairs of elements describe a step index
                    and the learning rate for that step and subsequent steps
                    until a new learning rate is specified
    last_epoch  -- the last run step
    named_lr_muls -- dictionary mapping names to multipliers on learning
                            rates specified in lrs. This is a simple and convenient way to have different learning rates for different layers
    """
    def __init__(self, optimizer, lrs, last_epoch=-1, named_lr_muls={}):
        super(StepScheduler, self).__init__()
        self.optimizer = optimizer
        self.named_lr_muls = named_lr_muls
        
        # Get a mapping from epoch indices to the learning rates they should if
        # the learning rate should change at the start of the epoch
        keys = [lrs[idx] for idx in range(0, len(lrs) -1, 2)]
        vals = [lrs[idx] for idx in range(1, len(lrs), 2)]
        self.schedule = OrderedDict(list(zip(keys, vals)))

        # Create a dictionary that implements (a) a fast mapping from steps to
        # the learning rate they should have, and (b) support for infinite steps
        # using the last learning rate
        self.step2lr = defaultdict(lambda: self.schedule[max(self.schedule.keys())])
        self.step2lr[-1] = self.schedule[0]
        cur_lr = self.schedule[0]
        for s in range(max(self.schedule.keys())):
            if s in self.schedule:
                cur_lr = self.schedule[s]
            self.step2lr[s] = cur_lr

        self.cur_step = last_epoch    
        self.step()
    
    def __str__(self): return f"{self.__class__.__name__} [schedule={dict(self.schedule)} cur_step={self.cur_step} lr={self.get_lr()}]"

    def get_lr(self): return self.step2lr[self.cur_step]
        
    def step(self, cur_step=None):
        cur_step = self.cur_step if cur_step is None else cur_step

        for pg in self.optimizer.param_groups:
            pg["lr"] = self.step2lr[cur_step]

            if "name" in pg and pg["name"] in self.named_lr_muls:
                pg["lr"] = pg["lr"] * self.named_lr_muls[pg["name"]]

        self.cur_step = cur_step + 1

    @staticmethod
    def process_lrs(lrs):
        """Returns a list where even elements give a step index and are integers
        and odd elements give the float learning rate starting at the prior even
        element step.

        This is intended to be run on the initial float-valied LRS attribute
        collected through argparse, and will raise argparse errors if the LRS
        specification is bad.
        """
        lrs = [float(l) for l in lrs]
        def is_increasing(l): return sorted(l) == l and len(l) == len(set(l))

        if not len(lrs) % 2 == 0:
            raise argparse.ArgumentTypeError(f"--lrs must have an even number of values")
        if not is_increasing([l for idx,l in enumerate(lrs) if idx % 2 == 0]):
            raise argparse.ArgumentTypeError(f"--lrs must have strictly increasing keys (even values)")
        if not lrs[0] == int(0):
            raise argparse.ArgumentTypeError(f"--lrs should begin with 0")
        else:
            return [int(l) if idx % 2 == 0 else float(l)
                for idx,l in enumerate(lrs)]




























