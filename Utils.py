import argparse
from collections import OrderedDict
from copy import deepcopy
import io
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import functional as tv_functional
import wandb


from tqdm import tqdm

# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import Misc

# I don't think this does anything because we don't have convolutions, but it
# probably can't hurt
torch.backends.cudnn.benchmark = False

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

def sample_latent_dict(d, bs=1, device=device, noise="gaussian", args=None):
    """Returns dictionary [d] after mapping all its values that are tuples of
    integers to Gaussian noise tensors with shapes given by the tuples.

    The noise and batch size specified as function parameters are default.
    However, they can be overridden on a per-key basis if needed. Example:
    ```
    {key: (1701,), key_bs: 42, key_noise_type: "zeros"}
    ```

    Args:
    d       -- a mapping from strings to dimensions, possibly with other
                key-value pairs
    bs      -- the batch size to use for each tensor if 'k_bs' isn't in [d]
    device  -- the device to return all tensors on
    noise   -- noise type that overrides args.noise
    args    -- Namespace with --noise (default noise that can be overrriden),
                --fix_mask_noise and --seed parameters
    """
    def get_sample(key, dims):
        if dims is None:
            dims = ()

        default_noise = args.noise if noise is None else noise
        noise_ = d[f"{key}_noise_type"] if f"{key}_noise_type" in d else default_noise
        bs_ = d[f"{key}_bs"] if f"{key}_bs" in d else bs

        # Sometimes for the masks we want to fix the latent seed
        if "mask" in key and not args is None and args.fix_mask_noise:
            gen = torch.Generator(device=device).manual_seed(args.seed)
        else:
            gen = None

        if noise_ == "gaussian":
            return torch.ones(*((bs_,) + dims), device=device).normal_(generator=gen)
        elif noise_ == "ones":
            return torch.ones(*((bs_,) + dims), device=device)
        elif noise_ == "zeros":
            return torch.zeros(*((bs_,) + dims), device=device)
        elif noise_ == "uniform":
            return torch.ones(*((bs_,) + dims), device=device).uniform_(generator=gen)            
        else:
            raise NotImplementedError(f"Unknown noise '{noise_}'")

    return {k: get_sample(k, v) for k,v in d.items()
        if not k.endswith("_noise_type") and not k.endswith("_bs")}

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
            self.idxs = Misc.sample(idxs, k=len(idxs), seed=seed)
        else:
            self.idxs = idxs

    def pop(self):
        if self.counter == len(self.idxs):
            self.counter = 0
            self.num_resets += 1
            if self.shuffle:
                self.idxs = Misc.sample(self.idxs,
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

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class NoChangeScheduler(_LRScheduler):
    """Scheduler that does not change learning rate.
    
    This scheduler will be the only thing to control the learning rate when in
    use, and the learning rate in the optimizer is ignored. This is a much more
    convenient API than allowing the optimizer to control the learning rate.
    """
    def __init__(self, optimizer, last_epoch=-1):
        self.global_step = last_epoch
        super(NoChangeScheduler, self).__init__(optimizer,
            last_epoch=last_epoch)
    
    def get_lr(self): return {pg["name"]: pg["lr"] for pg in self.optimizer.param_groups}

    def __str__(self): return f"{self.__class__.__name__} [{self.get_lr()}]"

    def step(self): self.global_step += 1


class LinearRampScheduler(_LRScheduler):
    """Scheduler giving a linear ramp followed by a constant learning rate.

    This scheduler will be the only thing to control the learning rate when in
    use, and the learning rate in the optimizer is ignored. This is a much more
    convenient API than allowing the optimizer to control the learning rate.

    Args:
    optimizer       -- wrapped optimizer
    warmup_steps    -- number of warmup steps for all parameters
    last_epoch      -- last epoch (step)
    min_lr          -- minimum learning rate for all parameters
    pg2base_lrs     -- dictionary mapping param group names in [optimizer] to
                        their post-ramp learning rates. If a float, the same
                        value is used for all parameters.
    pg2start_step   -- dictionary mapping each param group to the epoch on which
                        its learning rate can start being non-zero
    """
    def __init__(self, optimizer, warmup_steps=0, last_epoch=-1, min_lr=0,
        pg2base_lrs=1e-3, pg2start_step=None, verbose=True):

        if warmup_steps == 0:
            raise ValueError(f"LinearRampScheduler requires 'warmup_steps' > 0")
        self.global_step = last_epoch
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

        # Index of step on which a parameter group can start having a non-zero
        # learning
        if pg2start_step is None:
            self.pg2start_step = {p["name"]: 0 for p in optimizer.param_groups}
        else:
            self.pg2start_step = pg2start_step

        # Number of steps for each parameter group so far
        if last_epoch == -1:
            self.pg2step = {pg["name"]: 1 for pg in optimizer.param_groups}
        else:
            self.pg2step = {pg["name"]: max(1, (last_epoch - pg2start_step[pg["name"]]))
                for pg in optimizer.param_groups}


        # Amount to increase the learning rate of each parameter group per step
        # during its ramp
        if isinstance(pg2base_lrs, float):
            self.lr_inc_per_step = {pg["name"]: pg2base_lrs - min_lr
                for pg in optimizer.param_groups}
        elif isinstance(pg2base_lrs, dict):
            self.lr_inc_per_step = {pg["name"]: pg2base_lrs[pg["name"]] - min_lr
                for pg in optimizer.param_groups}
        else:
            raise NotImplementedError()

        self.lr_inc_per_step = {pgn: inc / warmup_steps
            for pgn,inc in self.lr_inc_per_step.items()}

        super(LinearRampScheduler, self).__init__(optimizer,
            last_epoch=-1, # Prevent the base class from being weird
            verbose=verbose)

    def get_lr(self): return {pg["name"]: pg["lr"] for pg in self.optimizer.param_groups}

    def step(self):
        for pg in self.optimizer.param_groups:
            if self.global_step < self.pg2start_step[pg["name"]]:     
                pg["lr"] = 0
            else:
                if self.pg2step[pg["name"]] <= self.warmup_steps:
                    pg["lr"] = self.pg2step[pg["name"]] * self.lr_inc_per_step[pg["name"]] + self.min_lr
                    self.pg2step[pg["name"]] += 1
                else:
                    self.pg2step[pg["name"]] += 1

        self.global_step += 1

    def __str__(self): return f"LinearRampScheduler [global_step {self.global_step} | min_lr {self.min_lr} | warmup_steps {self.warmup_steps}\n\tpg2step {self.pg2step} | pg2start_step {self.pg2start_step}\n\tlr_inc_per_step {self.lr_inc_per_step}\n\tlrs {self.get_lr()}]"

class LinearRampCosineDecayScheduler(_LRScheduler):

    def __init__(self, optimizer, warmup_steps=0, total_steps=1, last_epoch=-1, min_lr=0,
        pg2base_lrs=1e-3, pg2start_step=None, verbose=True):

        if warmup_steps == 0:
            raise ValueError(f"LinearRampCosineDecayScheduler requires 'warmup_steps' > 0")
        self.global_step = last_epoch
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Index of step on which a parameter group can start having a non-zero
        # learning
        if pg2start_step is None:
            self.pg2start_step = {p["name"]: 0 for p in optimizer.param_groups}
        else:
            self.pg2start_step = pg2start_step

        # Number of steps for each parameter group so far
        if last_epoch == -1:
            self.pg2step = {pg["name"]: 1 for pg in optimizer.param_groups}
        else:
            self.pg2step = {pg["name"]: max(1, (last_epoch - pg2start_step[pg["name"]]))
                for pg in optimizer.param_groups}

        # Set base learning rates
        if isinstance(pg2base_lrs, float):
            self.pg2base_lrs = {pg["name"]: pg2base_lrs for pg in optimizer.param_groups}
        else:
            self.pg2base_lrs = pg2base_lrs

        # Amount to increase the learning rate of each parameter group per step
        # during its ramp
        if isinstance(pg2base_lrs, float):
            self.lr_inc_per_step = {pg["name"]: pg2base_lrs - min_lr
                for pg in optimizer.param_groups}
        elif isinstance(pg2base_lrs, dict):
            self.lr_inc_per_step = {pg["name"]: pg2base_lrs[pg["name"]] - min_lr
                for pg in optimizer.param_groups}
        else:
            raise NotImplementedError()

        self.lr_inc_per_step = {pgn: inc / warmup_steps
            for pgn,inc in self.lr_inc_per_step.items()}

        super(LinearRampCosineDecayScheduler, self).__init__(optimizer,
            last_epoch=-1, # Prevent the base class from being weird
            verbose=verbose)

    def get_lr(self): return {pg["name"]: pg["lr"] for pg in self.optimizer.param_groups}

    def step(self):
        for pg in self.optimizer.param_groups:
            if self.global_step < self.pg2start_step[pg["name"]]:     
                pg["lr"] = 0
            else:
                if self.pg2step[pg["name"]] <= self.warmup_steps:
                    pg["lr"] = self.pg2step[pg["name"]] * self.lr_inc_per_step[pg["name"]] + self.min_lr
                    self.pg2step[pg["name"]] += 1
                else:
                    cosine_step_idx =  self.pg2step[pg["name"]] + self.pg2start_step[pg["name"]]
                    cosine_scaling = .5 + .5 * math.cos(math.pi * cosine_step_idx / self.total_steps)
                    pg["lr"] = cosine_scaling * self.pg2base_lrs[pg["name"]]
                    self.pg2step[pg["name"]] += 1

        self.global_step += 1

    def __str__(self): return f"{self.__class__.__name__} [global_step={self.global_step} min_lr={self.min_lr} warmup_steps={self.warmup_steps}\n\tpg2step={self.pg2step} pg2start_step={self.pg2start_step}\n\tlr_inc_per_step={self.lr_inc_per_step}\n\tlrs={self.get_lr()}]"


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




























