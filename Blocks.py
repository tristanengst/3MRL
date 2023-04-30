from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm

class Affine(nn.Module):
    """Applies a learned affine transformation over the last axis of the
    input.

    Args:
    dim     -- dimensionality of the last axis of the input
    weight  -- 
    bias    --
    """
    def __init__(self, dim=None, weight=None, bias=None):
        super(Affine, self).__init__()
        self.weight = nn.Parameter(torch.randn(dim)) if weight is None else nn.Parameter(weight.squeeze().detach())
        self.bias = nn.Parameter(torch.randn(dim)) if bias is None else nn.Parameter(bias.squeeze().detach())
        self.dim = self.bias.shape[0]

    def __str__(self): return f"{self.__class__.__name__} [dim={self.dim}]"

    def forward(self, x):
        return x * self.weight + self.bias

    @staticmethod
    def from_layernorm(l):
        """Returns an Affine layer with the parameters of LayerNorm [l]."""
        return Affine(weight=l.weight.detach(), bias=l.bias.detach())

    @staticmethod
    def make_block_start_with_affine(b):
        """Returns transformer block [b] with its first LayerNorm an equivalent
        Affine layer.
        """
        if isinstance(b, IPBlock):
            b.block.norm1 = Affine.from_layernorm(b.block.norm1)
        else:
            b.norm1 = Affine.from_layernorm(b.norm1)
        return b

class Block_(nn.Module):

    def __init__(self, block):
        super(Block_, self).__init__()
        self.block = block

    def forward(self, x, *args, **kwargs): return self.block(x)

class IPMethod(nn.Module):
    """Method for mapping noise to latent codes and fusing it with a
    representation.
    """
    def __init__(self, code_dim=512):
        super(IPMethod, self).__init__()
        self.code_dim = code_dim

    def get_latent_codes(self, bs=1, seed=None, device="cuda"):
        x = torch.zeros(bs, self.code_dim, device=device)
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        x.normal_(generator=generator)
        return x

    def forward(self, x): raise NotImplementedError()

def get_act(act_type):
    """Returns an activation function of type [act_type]."""
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=.2)
    else:
        raise NotImplementedError(f"Unknown activation '{act_type}'")

def get_lin_layer(in_dim, out_dim, equalized_lr=True, bias=True, **kwargs):
    """
    """
    if equalized_lr:
        return EqualizedLinear(in_dim, out_dim, bias=bias, **kwargs)
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)


class PixelNormLayer(nn.Module):
    """From https://github.com/huangzh13/StyleGAN.pytorch/blob/b1dfc473eab7c1c590b39dfa7306802a0363c198/models/CustomLayers.py.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier.
    
    From https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/CustomLayers.py.
    """

    def __init__(self, input_size, output_size, gain=2 ** .5, use_wscale=True, lrmul=.01, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias * self.b_mul if self.bias is not None else self.bias
        return nn.functional.linear(x, self.weight * self.w_mul, bias)

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim=256, out_dim=42, layers=4, 
        act_type="leakyrelu", equalized_lr=True, end_with_act=True, lrmul=1):
        super(MLP, self).__init__()

        if layers == 1 and end_with_act:
            self.model = nn.Sequential(
                get_lin_layer(in_dim, out_dim, equalized_lr=equalized_lr, lrmul=lrmul),
                get_act(act_type))
        elif layers == 1 and not end_with_act:
            self.model = get_lin_layer(in_dim, out_dim,
                equalized_lr=equalized_lr, lrmul=lrmul)
        elif layers > 1:
            layer1 = get_lin_layer(in_dim, h_dim, equalized_lr=equalized_lr, lrmul=lrmul, )
            mid_layers = [get_lin_layer(h_dim, h_dim, equalized_lr=equalized_lr, lrmul=lrmul)
                for _ in range(layers - 2)]
            layerN = get_lin_layer(h_dim, out_dim, equalized_lr=equalized_lr, lrmul=lrmul)
            linear_layers = [layer1] + mid_layers + [layerN]

            layers = []
            for idx,l in enumerate(linear_layers):
                layers.append(l)
                if end_with_act:
                    layers.append(get_act(act_type))
                elif not end_with_act and idx < len(linear_layers) - 1:
                    layers.append(get_act(act_type))
                else:
                    continue
            
            self.model = nn.Sequential(*layers)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
                
    def forward(self, x): return self.model(x)

class AdaIN(IPMethod):

    def __init__(self, args, c=768):
        super(AdaIN, self).__init__(code_dim=args.latent_dim)
        self.args = args
        self.c = c
        self.normalize_z = args.normalize_z
        self.mapping_net_h_dim = args.mapping_net_h_dim
        self.mapping_net_layers = args.mapping_net_layers
        self.latent_dim = args.latent_dim
        self.adain_x_norm = args.adain_x_norm

        layers = []
        if args.normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=0)))
        layers.append(("mapping_net", MLP(in_dim=args.latent_dim,
            h_dim=args.mapping_net_h_dim,
            layers=args.mapping_net_layers,
            out_dim=c * 2,
            act_type=args.mapping_net_act,
            equalized_lr=args.mapping_net_eqlr,
            lrmul=args.mapping_net_lrmul,
            end_with_act=False)))

        self.model = nn.Sequential(OrderedDict(layers))

        if args.adain_x_mod == "linear":
            self.x_modification_layer = nn.Linear(self.c, self.c)
        else:
            self.x_modification_layer = nn.Identity()
            
        if args.adain_x_norm == "none":
            self.x_norm_layer = nn.Identity()
        elif args.adain_x_norm == "norm":
            self.x_norm_layer = NormLayer()
        else:
            raise NotImplementedError()

        self.register_buffer("z_scale_mean", torch.zeros(self.c))
        self.register_buffer("z_shift_mean", torch.zeros(self.c))
        self.init_constants()

    def get_z_stats(self, num_z=2048, device="cpu"):
        """Returns the mean shift and scale used in the AdaIN, with the mean
        taken over [num_z] different latent codes.
        """
        with torch.no_grad():
            z = self.model(get_codes(num_z, self.latent_dim, device=device))
            z_shift, z_scale = z[:, :self.c], z[:, self.c:]
            return torch.mean(z_shift, dim=0), torch.std(z_shift, dim=0), torch.mean(z_scale, dim=0), torch.std(z_scale, dim=0)

    def init_constants(self, num_z=2048):
        """Sets the [z_shift_mean] and [z_shift_scale] constants."""
        self.z_shift_mean, _, self.z_scale_mean, _ = self.get_z_stats(num_z=num_z)

    def forward(self, x, latent_codes=None, seed=None, codes_per_ex=1, ignore_z=False):
        """
        Args:
        x   -- image features
        z   -- latent codes or False for acting like an identity function
        """
        if ignore_z:
            return torch.repeat_interleave(x, codes_per_ex, dim=0) 

        if latent_codes is None:
            latent_codes = self.get_latent_codes(bs=len(x) * codes_per_ex, seed=seed,
                device=x.device)
        
        z = self.model(latent_codes)
        z_shift = z[:, :self.c] - self.z_shift_mean
        z_scale = z[:, self.c:] - self.z_scale_mean

        z_scale = torch.nn.functional.relu(z_scale)
        x = self.x_modification_layer(x)
        x = self.x_norm_layer(x)

        x = torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)
        z_shift = z_shift.unsqueeze(1).expand(*x.shape)
        z_scale = z_scale.unsqueeze(1).expand(*x.shape)
        result = z_shift + x * (1 + z_scale)
        return result

class AdaIN_(IPMethod):
    """AdaIN adapted for a transformer. Expects a BSxNPxC batch of images, where
    each image is represented as a set of P tokens, and BSxPxZ noise. This noise
    is mapped to be BSx1x2C. These are used to scale the image patches, ie. in
    the ith image, the kth element of the jth patch is scaled identically to the
    kth element of any other patch in that image.
    """
    def __init__(self, c, epsilon=1e-8, act_type="leakyrelu", normalize_z=True):
        super(AdaIN_, self).__init__(code_dim=512)
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.c = c
        
        layers = []
        if normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=epsilon)))
        
        layers.append(("mapping_net", MLP(in_dim=512,
            h_dim=512,
            layers=8,
            out_dim=self.c * 2,
            equalized_lr=True,
            act_type=act_type)))

        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x, latent_codes=None, seed=None, codes_per_ex=1, ignore_z=False):
        """
        Args:
        x   -- image features
        z   -- latent codes or False for acting like an identity function
        """
        if ignore_z:
            return torch.repeat_interleave(x, codes_per_ex, dim=0) 

        if latent_codes is None:
            latent_codes = self.get_latent_codes(bs=len(x) * codes_per_ex, seed=seed,
                device=x.device)

        z = self.model(latent_codes)
        z_mean = z[:, :self.c]
        z_std = z[:, self.c:]

        x = torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)
        z_mean = z_mean.unsqueeze(1).expand(*x.shape)
        z_std = z_std.unsqueeze(1).expand(*x.shape)
        result = z_mean + x * (1 + z_std)
        return result

class LocalAdaIN(IPMethod):
    """AdaIN adapted for a transformer. Expects a BSxNPxC batch of images, where
    each image is represented as a set of P tokens, and BSxPxZ noise. This noise
    is mapped to be BSxNPx2. These are used to scale the image patches, ie. each
    patch is scaled and shifted by a (very likely) different amount.

    CAN NOT BE FOLLOWED BY LAYER NORM OR DOES PLAIN MAE.
    """
    def __init__(self, c, epsilon=1e-8, act_type="leakyrelu", normalize_z=True):
        super(LocalAdaIN, self).__init__(code_dim=512)
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.c = c

        layers = []
        if normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=epsilon)))
        layers.append(("mapping_net", MLP(in_dim=512,
            h_dim=512,
            layers=8,
            out_dim=self.c * 2,
            equalized_lr=True,
            act_type=act_type)))

        self.model = nn.Sequential(OrderedDict(layers))
        
    def forward(self, x, latent_codes=None, seed=None, codes_per_ex=1):
        """Returns the LocalAdaINification of [x] given codes [z]. As the
        network assumes a fixed number of patches, only cases where [x] has the
        right number will be scaled; otherwise this functions like an identity
        function.

        Args:
        x   -- image features
        z   -- latent codes
        """
        if ignore_z:
            x = torch.repeat_interleave(x, codes_per_ex, dim=0) 

        if latent_codes is None:
            latent_codes = self.get_latent_codes(bs=len(x) * codes_per_ex, seed=seed,
                device=x.device)
        
        
        z = self.model(latent_codes)
        z_mean = z[:, :self.c]
        z_std = z[:, self.c:]

        x = torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)
        z_mean = z_mean.unsqueeze(1).expand(*x.shape)
        z_std = z_std.unsqueeze(1).expand(*x.shape)
        result = z_mean + x * (1 + z_std)
        return result

def get_codes(bs, code_dim, device="cpu", seed=None):
    """Returns [bs] latent codes to be passed into the model.

    Args:
    bs          -- number of latent codes to return
    code_dim    -- dimension of input latent codes
    device      -- device to return latent codes on
    seed        -- None for no seed (outputs will be different on different
                    calls), or a number for a fixed seed
    """
    if seed is None:
        return torch.randn(bs, code_dim, device=device)
    else:
        z = torch.zeros(bs, code_dim, device=device)
        z.normal_(generator=torch.Generator(device).manual_seed(seed))
        return z

