import argparse
import io
import matplotlib
from ApexUtils import *
from torchvision.transforms import functional as tv_functional

# I don't think this does anything because we don't have convolutions, but it
# probably can't hurt
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["savefig.bbox"] = "tight"

project_dir = f"{os.path.dirname(__file__)}"
data_dir = f"{project_dir}/data"

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

def parse_variational_spec(args):
    """Returns args.v_spec as a dictionary mapping transformer block indices to
    whether and how they should be variational. Blocks whose indices aren't in
    the mapping are assumed in model __init__ methods to be non-variational and
    such blocks need not be specified in the returned dictionary.
    """
    def parse_v_spec_helper(s):
        if s in ["add", "zero"] or not s:
            return s
        elif s.startswith("downsample_mlp_"):
            hidden_dim = int(s[len("downsample_mlp_"):])
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    key_args = [int(k) for idx,k in enumerate(args.v_spec) if idx % 2 == 0]
    val_args = [v for idx,v in enumerate(args.v_spec) if idx % 2 == 1]
    assert len(key_args) == len(val_args)
    return {k: parse_v_spec_helper(v) for k,v in zip(key_args, val_args)}

def sample_latent_dict(d, bs=1, device=device, noise="gaussian"):
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
    noise   -- the type of noise if for key [k] if 'k_noise_type' isn't in [d]
    """
    def get_sample(key, dims):
        noise_ = d[f"{key}_noise_type"] if f"{key}_noise_type" in d else noise
        bs_ = d[f"{key}_bs"] if f"{key}_bs" in d else bs

        if noise_ == "gaussian":
            return torch.randn(*((bs_,) + dims), device=device)
        elif noise_ == "ones":
            return torch.ones(*((bs_,) + dims), device=device)
        elif noise_ == "zeros":
            return torch.zeros(*((bs_,) + dims), device=device)
        else:
            raise NotImplementedError(f"Unknown noise '{noise}'")

    return {k: get_sample(k, v) for k,v in d.items()
        if not k.endswith("_noise_type") and not k.endswith("_bs")}

def images_to_pil_image(images):
    """Returns tensor datastructure [images] as a PIL image."""
    images = make_2d_list_of_tensor(images)

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
