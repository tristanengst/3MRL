import argparse
from ApexUtils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

project_dir = f"{os.path.dirname(__file__)}"
data_dir = f"{project_dir}/data"

def sample_latent_dict(d, bs, device=device, noise="gaussian"):
    """Returns dictionary [d] after mapping all its values that are tuples of
    integers to Gaussian noise tensors with shapes given by the tuples.
    """
    if isinstance(d, dict) and "shape" and "batch_dim" in d:
        s = d["shape"][:d["batch_dim"]] + (bs,) + d["shape"][d["batch_dim"]:]
        if noise == "gaussian":
            return torch.randn(*s, device=device)
        elif noise == "ones":
            return torch.ones(*s, device=device)
    elif isinstance(d, dict):
        return {k: sample_latent_dict(v, bs, noise=noise) for k,v in d.items()}
    else:
        raise NotImplementedError()

def namespace_with_update(args, key, value):
    """Returns a Namespace identical to [args] but with [key] set to [value]."""
    return argparse.Namespace(**(vars(args) | {new_param: value}))