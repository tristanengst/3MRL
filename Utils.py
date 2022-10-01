import argparse
import io
# from PIL import Image
import matplotlib
from ApexUtils import *
from torchvision.transforms import functional as tv_functional

device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["savefig.bbox"] = "tight"

project_dir = f"{os.path.dirname(__file__)}"
data_dir = f"{project_dir}/data"

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
        if s in ["add"] or not s:
            return s
        elif s.starswith("downsample_mlp_"):
            hidden_dim = int(s[len("downsample_mlp_"):])
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    key_args = [int(k) for idx,k in enumerate(args.v_spec) if idx % 2 == 0]
    val_args = [v for idx,v in enumerate(args.v_spec) if idx % 2 == 1]
    return {k: parse_v_spec_helper(v) for k,v in zip(key_args, val_args)}

def sample_latent_dict(d, bs, device=device, noise="gaussian", key=None):
    """Returns dictionary [d] after mapping all its values that are tuples of
    integers to Gaussian noise tensors with shapes given by the tuples.

    Args:
    d       -- a mapping from strings to tuples giving the shapes of noise
                tensors to create
    bs      -- integer giving the batch size (appended zero dimension) for all
                returned tensors, or a mapping from keys in [d] to the required
                batch sizes
    device  -- the device to return tensors on
    noise   -- the kind of noise to add
    key     -- the key associated with a value in [d]. Used recursively only
    """
    if isinstance(d, tuple):
        bs = bs[key] if isinstance(bs, dict) and key in bs else bs
        s = (bs,) + d
        if noise == "gaussian":
            return torch.randn(*s, device=device)
        elif noise == "ones":
            return torch.ones(*s, device=device)
        elif noise == "zeros":
            return torch.zeros(*s, device=device)
        else:
            raise NotImplementedError()
    elif d is None:
        return None
    elif isinstance(d, dict):
        return {k: sample_latent_dict(v, bs, noise=noise, key=k)
            for k,v in d.items()}
    else:
        raise NotImplementedError()

def namespace_with_update(args, key, value):
    """Returns a Namespace identical to [args] but with [key] set to [value]."""
    return argparse.Namespace(**(vars(args) | {new_param: value}))

def images_to_pil_image(images):
    """Returns tensor datastructure [images] as a PIL image."""
    images = make_2d_list_of_tensor(images)

    fig, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images),
        squeeze=False)

    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            image = np.asarray(tv_functional.to_pil_image(image.detach()))
            axs[i, j].imshow(image, cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    plt.close("all")
    return Image.open(buf)

def show_image_grid(images):
    """Shows list of images [images], either a Tensor giving one image, a List
    where each element is a Tensors giving one images, or a 2D List where each
    element is a Tensor giving an image.
    """
    images = make_2d_list_of_tensor(images)


    fig, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images), squeeze=False)


    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            axs[i, j].imshow(np.asarray(functional_TF.to_pil_image(image.detach())), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
