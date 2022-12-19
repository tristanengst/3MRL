"""

NOTES:
1. Pretrained models are available at https://github.com/facebookresearch/mae/issues/8, which isn't the same as those directly linked in the repo.
"""
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block
from original_code.models_vit import VisionTransformer

from original_code.util import pos_embed as PE

from Utils import *
from Blocks import AdaIN, LocalAdaIN
from Augmentation import de_normalize

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


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024,
        depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8,
        decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
        norm_pix_loss=False, verbose=False):
        super(MaskedAutoencoderViT, self).__init__()

        # Save variables needed in making this IP
        self.kwargs = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_chans": in_chans,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "decoder_embed_dim": decoder_embed_dim,
            "decoder_depth": decoder_depth,
            "decoder_num_heads": decoder_num_heads,
            "mlp_ratio": mlp_ratio,
            "norm_layer": norm_layer,
            "norm_pix_loss": norm_pix_loss,
        }

        self.encoder_kwargs = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_chans": in_chans,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "norm_layer": norm_layer,
        }

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        if verbose:
            tqdm.write(f"{self.__class__.__name__} [num_blocks {len(self.blocks)} | num_params {sum(p.numel() for p in self.parameters() if p.requires_grad)}]")

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = PE.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = PE.get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], 3, h * p, h * p))

    def random_masking(self, x, mask_ratio, mask_noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        Args:
        x           -- input to mask
        mask_ratio  -- fraction to mask
        mask_noise  -- noise to construct the mask with or None to generate it
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        if mask_noise is None:  # batch, length, dim
            mask_noise = torch.rand(N, L, device=x.device)
        else:
            mask_noise = mask_noise
            
        ids_shuffle = torch.argsort(mask_noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        ids_restore = ids_restore.repeat_interleave(len(x) // len(ids_restore), dim=0)

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask, reduction="mean", mask_ratio=1):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        target = target.repeat_interleave(len(pred) // len(target), dim=0)
        mask = mask.repeat_interleave(len(pred) // len(mask), dim=0)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss * mask

        if reduction == "mean":
            return (loss.sum() / mask.sum()).unsqueeze(0)  # The 0-1ness of [mask] makes this not dependent on batch size
        elif reduction == "batch":
            return torch.mean(loss, axis=1) / mask_ratio
        else:
            raise NotImplementedError()

    def forward(self, x, z=None, mask_ratio=0.75, reduction="mean", return_all=False):
        """
        Args:
        x           -- BSxCxHxW tensor of images to encode
        mask_ratio  -- mask ratio
        """
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask,
            reduction=reduction,
            mask_ratio=mask_ratio)
        
        if return_all:
            pred, mask = restore_model_outputs(pred, mask, self.patch_embed,
                self.unpatchify)
            return loss, pred, mask
        else:
            return loss

    def get_latent_spec(self, test_input=None, input_size=None, mask_ratio=.75):
        """Returns the latent specification for the network. Requires that the
        model is on the "cuda" device.

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        input_size  -- spatial resolution of a tensor to be used in place of
                        [test_input] if [test_input] is None
        mask_ratio  -- mask ratio
        """
        if not input_size is None and test_input is None:
            test_input = torch.ones(4, 3, input_size, input_size, device="cuda")

        with torch.no_grad():
            x = self.patch_embed(test_input)
            x = x + self.pos_embed[:, 1:, :]
            mask_noise_shape = (x.shape[1],)
            return {"mask_noise": mask_noise_shape,
                "mask_noise_type": "uniform",
                "latents": None}

class VisionTransformerBackbone(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super(VisionTransformerBackbone, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward_features(*args, **kwargs)

def parse_ip_spec(args):
    """Returns args.ip_spec as a dictionary mapping transformer block indices to
    whether and how they should be IP. Blocks whose indices aren't in
    the mapping are assumed in model __init__ methods to be non-IP and
    such blocks need not be specified in the returned dictionary.
    """
    def parse_ip_spec_helper(s):
        if s in ["add", "zero"] or not s:
            return s
        elif s.startswith("adain"):
            if "base" in args.arch:
                return AdaIN(c=768, act_type=args.act_type)
            elif "large" in args.arch:
                return AdaIN(c=1024, act_type=args.act_type)
            else:
                raise NotImplementedError()
        elif s.startswith("local_adain"):
            return LocalAdaIN(c=50, act_type=args.act_type)
        else:
            raise NotImplementedError()

    key_args = [int(k) for idx,k in enumerate(args.ip_spec) if idx % 2 == 0]
    val_args = [v for idx,v in enumerate(args.ip_spec) if idx % 2 == 1]
    assert len(key_args) == len(val_args)
    return {k: parse_ip_spec_helper(v) for k,v in zip(key_args, val_args)}


def extend_idx2ip_method(idx2ip_method, length):
    """Returns [idx2ip_method] extended to [length] by adding key-value pairs
    where the value is False such that the returned dictionary has the first
    [length] whole numbers as keys.

    Args:
    idx2ip_method    -- mapping from the first N whole numbers to how/if blocks
                        with their index should be IP
    length          -- length to extend to
    """
    return {k: False for k in range(length)} | idx2ip_method


class IPBlock(nn.Module):
    """Wraps a vision transformer block and adds noise to its output.
    Args:
    block       -- the vision transformer block to wrap
    ip_method    -- how to add noise (a) 'add' to have the noise added to the
                    block's output, or (b) an nn.Module taking the output and
                    the noise as input and producing a new output with the same
                    dimenion as the block's output
    """
    def __init__(self, block, ip_method="add"):
        super(IPBlock, self).__init__()
        self.block = block
        self.ip_method = ip_method

    def get_latent_spec(self, test_input):
        """Returns the latent shape minus the batch dimension for the network.

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        """
        block_output = self.block(test_input)
        if self.ip_method == "add":
            return block_output.shape[1:]
        elif isinstance(self.ip_method, nn.Module):
            return self.ip_method.get_latent_spec(block_output)
        else:
            raise NotImplementedError()

    def forward(self, x, z, ignore_z=False):
        """Forward function.

        Args:
        x   -- input to the wrapped block
        z   -- either a tensor to use as noise or an nn.Module whose forward
                function returns the noise. One might use the former type for
                IMLE training and the latter for density-based linear probing
        """
        fx = self.block(x)
        if ignore_z:
            return torch.repeat_interleave(fx, z.shape[0] // fx.shape[0], dim=0)

        z = z() if isinstance(z, nn.Module) else z
        if self.ip_method == "add":
            fx = torch.repeat_interleave(fx, z.shape[0] // fx.shape[0], dim=0)
            return fx + z
        elif isinstance(self.ip_method, nn.Module):
            return self.ip_method(fx, z)
        else:
            raise NotImplementedError()

class IPViT(timm.models.vision_transformer.VisionTransformer):
    """ViT, but with the ability to add Gaussian noise at some place in the
    forward pass.

    To construct from a MaskedIPViT model [m]:
    v = IPViT(idx2ip_method=m.idx2ip_method,
        encoder_kwargs=m.encoder_kwargs, ...)
    v.load_state_dict(m.state_dict())

    Args:
    idx2ip_method    -- mapping from from indices to the blocks to whether and
                        how they should be IP. If the ith value in the
                        mapping is False, the ith block will not be made
                        IP. Otherwise, it will be replaced with a
                        IPBlock using the value as [ip_method].

                        All missing key-value pairs will be added with the value
                        set to False.
    encoder_kwargs  -- encoder_kwargs of a MaskedIPAutoencoder model.
                        overrides [kwargs] on conflicting entries.
    global_pool     -- use global pooling or cls token for representations
    num_classes     -- number of classes for classification
    kwargs          -- arguments for constructing the ViT architecture
    """
    def __init__(self, idx2ip_method={}, encoder_kwargs=None, global_pool=False,
        num_classes=1000, noise="gaussian", **kwargs):

        # The architecture [v_mae_model] overrides that in [kwargs] if possible
        kwargs = kwargs | {"num_classes": num_classes}
        kwargs = kwargs if encoder_kwargs is None else kwargs | encoder_kwargs
        super(IPViT, self).__init__(**kwargs)

        self.idx2ip_method = extend_idx2ip_method(idx2ip_method, len(self.blocks))
        
        ########################################################################
        # Make the LayerNorms starting a block an Affine layer if the block is
        # preceeded by a LocalAdaIN
        idx2block = []
        for idx,b in enumerate(self.blocks):
            prev_block_idx = idx - 1
            if (prev_block_idx in self.idx2ip_method
                and isinstance(self.idx2ip_method[prev_block_idx], LocalAdaIN)):
                idx2block.append(Affine.make_block_start_with_affine(b))
            else:
                idx2block.append(b)

        # Make some blocks implicit probabilistic
        idx2block = [IPBlock(b, vm) if vm else b
            for b,vm in zip(idx2block, self.idx2ip_method.values())]
        self.idx2block = {str(idx): b for idx,b in enumerate(idx2block)}
        self.idx2block = nn.ModuleDict(self.idx2block)
        del self.blocks

        # Get the mapping for latent codes being put into blocks
        self.block_idx2z_idx = {b_idx: z_idx
            for z_idx,b_idx in enumerate([
                b_idx for b_idx,vm in self.idx2ip_method.items() if vm])}

        ########################################################################

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm            

        self.latent_spec = None
        self.latent_spec_test_input_dims = None
        self.latent_spec_mask_ratio = None

        self.noise = noise

    def set_latent_spec(self, test_input=None, input_size=None, mask_ratio=0):
        """Sets the three instance variables storing the latent spec. Requires
        that the model is on the "cuda" device.

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        input_size  -- spatial resolution of a tensor to be used in place of
                        [test_input] if [test_input] is None
        mask_ratio  -- mask ratio
        """
        if (mask_ratio == self.latent_spec_mask_ratio
            and test_input.shape == self.latent_spec_test_input_dims
            and not self.latent_spec is None):
            return None
        else:
            self.latent_spec = self.get_latent_spec(test_input=test_input,
                input_size=input_size,
                mask_ratio=mask_ratio)
            self.latent_spec_test_input_dims = test_input.shape
            self.latent_spec_mask_ratio = mask_ratio
            return None

    def get_latent_spec(self, test_input=None, input_size=None, mask_ratio=.75):
        """Returns the latent specification for the network. Requires that the
        model is on the "cuda" device.

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        input_size  -- spatial resolution of a tensor to be used in place of
                        [test_input] if [test_input] is None
        mask_ratio  -- mask ratio
        """
        if not input_size is None and test_input is None:
            test_input = torch.ones(4, 3, input_size, input_size, device="cuda")

        shapes = []
        with torch.no_grad():
            B = test_input.shape[0]
            x = self.patch_embed(test_input)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for idx,block in self.idx2block.items():
                if int(idx) in self.block_idx2z_idx:
                    s = block.get_latent_spec(test_input=x)
                    x = block(x, torch.ones(len(x), *s, device="cuda"))
                    shapes.append(s)
                else:
                    x = block(x)

        if len(set(shapes)) == 0:
            return {"latents": None}
        elif len(set(shapes)) == 1:
            return {"latents": (len(shapes),) + shapes[0]}
        else:
            raise ValueError(f"Got multiple shapes for latent codes {shapes}")

    def forward_features(self, x, z=None, z_per_ex=1, noise=None, ignore_z=False):
        """Returns representations for [x].

        Args:
        x   -- BSxCxHxW images to compute representations for
        z   -- list of noises, one for each IP block
        """
        if z is None:
            z = sample_latent_dict(self.latent_spec,
                bs=len(x) * z_per_ex,
                noise=self.noise if noise is None else noise)

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx,block in self.idx2block.items():
            if int(idx) in self.block_idx2z_idx:
                x = block(x, z["latents"][:, self.block_idx2z_idx[int(idx)]], ignore_z=ignore_z)
            else:
                x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            return self.fc_norm(x)
        else:
            x = self.norm(x)
            return x[:, 0]

    def forward(self, x, z=None, noise=None, z_per_ex=1, ignore_z=False):
        """Forward function for getting class predictions.

        Requires that set_latent_spec() has been called on the model.

        Args:
        x       -- BSxCxHxW tensor of images
        z       -- filled-in latent specification of size matching tha tof [x]
        noise   -- method for generating [z] if it is None
        z_per_x -- number of latents for example if [z] is None
        """
        return self.forward_head(self.forward_features(x, z,
            noise=noise, z_per_ex=z_per_ex, ignore_z=ignore_z))

class IPViTBackbone(IPViT):
    """IPViT as a backbone network."""
    def __init__(self, **kwargs):
        super(IPViTBackbone, self).__init__(**kwargs)

    def forward(self, *args, **kwargs):
        return super().forward_features(*args, **kwargs)

class MaskedIPViT(MaskedAutoencoderViT):
    """Masked VAE with VisionTransformer backbone.

    Args:
    idx2ip_method    --  mapping from from indices to the blocks to whether and
                        how they should be IP. If the ith value in the
                        mapping is False, the ith block will not be made
                        IP. Otherwise, it will be replaced with a
                        IPBlock using the value as [ip_method].

                        All missing key-value pairs will be added with the value
                        set to False.
    mae_model       -- None or MaskedAutoencoderViT instance. If the former,
                        values in [kwargs] are used to specify the ViT
                        architecture; if the latter, the architecture of
                        [mae] model is used to for this one
    **kwargs        -- same kwargs as for a MaskedAutoencoderViT. Ignored if
                        [mae_model] is specified.
    """
    def __init__(self, idx2ip_method={}, mae_model=None, **kwargs):
        if mae_model is None:
            super(MaskedIPViT, self).__init__(**kwargs)
        else:
            super(MaskedIPViT, self).__init__(**mae_model.kwargs)
            self.load_state_dict(mae_model.state_dict(), strict=False)

        self.idx2ip_method = extend_idx2ip_method(idx2ip_method, len(self.blocks))
        ########################################################################
        # Make the LayerNorms starting a block an Affine layer if the block is
        # preceeded by a LocalAdaIN
        idx2block = []
        for idx,b in enumerate(self.blocks):
            prev_block_idx = idx - 1
            if (prev_block_idx in self.idx2ip_method
                and isinstance(self.idx2ip_method[prev_block_idx], LocalAdaIN)):
                idx2block.append(Affine.make_block_start_with_affine(b))
            else:
                idx2block.append(b)

        # Make some blocks implicit probabilistic
        idx2block = [IPBlock(b, vm) if vm else b
            for b,vm in zip(idx2block, self.idx2ip_method.values())]
        self.idx2block = {str(idx): b for idx,b in enumerate(idx2block)}
        self.idx2block = nn.ModuleDict(self.idx2block)
        del self.blocks

        # Get the mapping for latent codes being put into blocks
        self.block_idx2z_idx = {b_idx: z_idx
            for z_idx,b_idx in enumerate([
                b_idx for b_idx,vm in self.idx2ip_method.items() if vm])}

        # If the last encoder block ends with a LocalAdaIN, the decoder norm and
        # the first norm of the first decoder block need to become Affine layers
        if isinstance(self.idx2ip_method[len(idx2block) - 1], LocalAdaIN):
            self.norm = Affine.from_layernorm(self.norm)
            self.decoder_blocks[0] = Affine.make_block_start_with_affine(self.decoder_blocks[0])
        ########################################################################

        tqdm.write(f"{self.__class__.__name__} [num_blocks {len(self.idx2block)} | num_params {sum(p.numel() for p in self.parameters() if p.requires_grad)}]")

    def get_latent_spec(self, test_input=None, input_size=None, mask_ratio=.75):
        """Returns the latent specification for the network. Requires that the
        model is on the "cuda" device.

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        input_size  -- spatial resolution of a tensor to be used in place of
                        [test_input] if [test_input] is None
        mask_ratio  -- mask ratio
        """
        if not input_size is None and test_input is None:
            test_input = torch.ones(4, 3, input_size, input_size, device="cuda")

        shapes = []
        with torch.no_grad():
            x = self.patch_embed(test_input)
            x = x + self.pos_embed[:, 1:, :]

            # Shape of noise needed for the mask
            mask_noise_shape = (x.shape[1],)

            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for idx,block in self.idx2block.items():
                if int(idx) in self.block_idx2z_idx:
                    s = block.get_latent_spec(test_input=x)
                    shapes.append(s)
                    x = block(x, torch.zeros(len(x), *s, device="cuda"))
                else:
                    x = block(x)

        if len(set(shapes)) == 0:
            return {"mask_noise": mask_noise_shape,
                "mask_noise_type": "uniform",
                "latents": None}
        elif len(set(shapes)) == 1:
            return {"mask_noise": mask_noise_shape,
                "mask_noise_type": "uniform",
                "latents": (len(shapes),) + shapes[0]}
        else:
            raise ValueError(f"Got multiple shapes for latent codes {shapes}")

    def forward_encoder(self, x, z, mask_ratio, ignore_z=False):
        """Forward function for the encoder.

        Args:
        x           -- BSxCxHxW tensor of images to encode
        z           -- dictionary with keys 'mask_noise' and 'latents'. Each
                        should map to a BSx... tensor with the non-zero
                        dimensions given by get_latent_spec().
        mask_ratio  -- mask ratio
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask_noise=z["mask_noise"])

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for idx,block in self.idx2block.items():
            if int(idx) in self.block_idx2z_idx:
                x = block(x, z["latents"][:, self.block_idx2z_idx[int(idx)]], ignore_z=ignore_z)
            else:
                x = block(x)

        x = self.norm(x)
        return x, mask, ids_restore

    def forward(self, x, z, mask_ratio=0.75, reduction="mean", return_all=False, ignore_z=False):
        """Forward function returning the loss, reconstructed image, and mask.

        Args:
        x           -- BSxCxHxW tensor of images to encode
        z           -- dictionary with keys 'mask_noise' and 'latents'. Each
                        should map to a BSx... tensor with the non-zero
                        dimensions given by get_latent_spec().
        mask_ratio  -- mask ratio
        reduction   -- reduction applied to returned loss
        return_all  -- whether to return a (loss, images, masks) tuple or loss
        """
        latent, mask, ids_restore = self.forward_encoder(x, z, mask_ratio,
            ignore_z=ignore_z)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask,
            reduction=reduction,
            mask_ratio=mask_ratio)
        
        if return_all:
            pred, mask = restore_model_outputs(pred, mask, self.patch_embed,
                self.unpatchify)
            return loss, pred, mask
        else:
            return loss

def restore_model_outputs(pred, mask, patch_embed, unpatchify):
    """Returns a (pred, mask) tuple after modifying each to be viewable.

    Args:
    pred        -- the prediction
    mask        -- the mask used to create [pred]
    patch_embed -- patch_embed attribute of the model outputting pred
    unpatchify  -- unpatchify function of the model outputting pred
    """
    pred = de_normalize(unpatchify(pred))
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_embed.patch_size[0]**2 *3)
    mask = unpatchify(mask)
    return pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def arch_to_model(args):
    """
    """
    if args.arch == "base_patch16":
        return mae_vit_base_patch16

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
