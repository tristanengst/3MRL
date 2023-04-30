"""

NOTES:
1. Pretrained models are available at https://github.com/facebookresearch/mae/issues/8, which isn't the same as those directly linked in the repo.
"""
from functools import partial
import os
from tqdm import tqdm

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block
from original_code.models_vit import VisionTransformer

from original_code.util import pos_embed as PE

from Utils import *
from Blocks import AdaIN, LocalAdaIN, MLP
from Augmentation import de_normalize

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
                return AdaIN(args, c=768)
            elif "large" in args.arch:
                return AdaIN(args, c=768)
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

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024,
        depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8,
        decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
        norm_pix_loss=False, verbose=False):
        super(MaskedAutoencoderViT, self).__init__()

        # Save variables needed in making this an implicit model
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

    def get_latent_codes(self, bs=1, device="cuda", seed=None):
        """Returns a size [bs] vector of "latents" to allow this model to
        pretend to be an implicit probabilistic model.
        """
        return torch.zeros(bs, device=device)

    def get_mask_codes(self, bs=1, device="cuda", seed=None):
        """Returns mask codes for [bs] images.
        
        Args:
        bs      -- the number of codes to generate
        device  -- the device to generate the codes on
        seed    -- the seed used to generate the random noise, or None to use
                    the current state of randomness
        """
        L = self.kwargs["img_size"] ** 2 // (self.kwargs["patch_size"] ** 2)
        mask_codes = torch.zeros(bs, L, device=device)
        g = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        mask_codes.normal_(generator=g)
        return mask_codes

    def random_masking(self, x, mask_ratio=.75, mask_codes=None, mask_seed=None):
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

        ########################################################################
        if mask_codes is None:
            mask_codes = self.get_mask_codes(bs=x.shape[0], device=x.device, seed=mask_seed)
        else:
            mask_codes = mask_codes
        ########################################################################
            
        ids_shuffle = torch.argsort(mask_codes, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask_codes=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask_codes=mask_codes)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        ids_restore = ids_restore.repeat_interleave(len(x) // len(ids_restore), dim=0)
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
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

    def forward(self, x, mask_ratio=0.75, reduction="mean", return_all=False):
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

class VisionTransformerBackbone(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super(VisionTransformerBackbone, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward_features(*args, **kwargs)

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

    def get_latent_codes(self, bs=1, device=device, seed=None):
        """Returns [bs] latent codes for the block on device [device] generated
        with seed [seed], or the current state of randomness if [seed] is None.
        """
        return self.ip_method.get_latent_codes(bs=bs, device=device, seed=seed)        

    def forward(self, x, latent_codes=None, ignore_z=False, codes_per_ex=1, seed=None):
        """Forward function.

        Args:
        x   -- input to the wrapped block
        z   -- either a tensor to use as noise or an nn.Module whose forward
                function returns the noise. One might use the former type for
                IMLE training and the latter for density-based linear probing
        ignore_z
        codes_per_ex
        seed
        """
        fx = self.block(x)
        return self.ip_method(fx, latent_codes=latent_codes, codes_per_ex=codes_per_ex, seed=seed, ignore_z=ignore_z)
            

class IPViT(timm.models.vision_transformer.VisionTransformer):
    """ViT with the ability to add Gaussian noise somewhere in the forward pass.

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

        kwargs = kwargs | {"num_classes": num_classes}
        kwargs = kwargs if encoder_kwargs is None else kwargs | encoder_kwargs
        super(IPViT, self).__init__(**kwargs)

        self.idx2ip_method = extend_idx2ip_method(idx2ip_method, len(self.blocks))
        
       ########################################################################
        # Make the LayerNorms starting a block an Affine layer if the block is
        # preceeded by a LocalAdaIN
        blocks = []
        for idx,b in enumerate(self.blocks):
            prev_block_idx = idx - 1
            if (prev_block_idx in self.idx2ip_method
                and isinstance(self.idx2ip_method[prev_block_idx], LocalAdaIN)):
                blocks.append(Affine.make_block_start_with_affine(b))
            else:
                blocks.append(b)

        # Make some blocks implicit probabilistic given the IP specification
        self.blocks = nn.ModuleList([IPBlock(b, ipm) if ipm else b for b,ipm in zip(blocks, self.idx2ip_method.values())])

        # Create a dictionary that gives the index to the latent codes for the
        # index of each index of the IP blocks
        counter = 0
        self.block_idx2latent_idx = {}
        for idx,ipm in self.idx2ip_method.items():
            if ipm:
                self.block_idx2latent_idx[idx] = counter
                counter += 1
        
        self.first_ip_idx = min(self.block_idx2latent_idx.keys())

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

    def get_latent_codes(self, bs, **kwargs):
        """Returns [bs] latent codes to be input to the model."""
        result = [block.get_latent_codes(bs, **kwargs)
            for idx,block in enumerate(self.blocks)
            if idx in self.block_idx2latent_idx]
        return torch.stack(result, dim=1)

    def forward_features(self, x, latent_codes=None, codes_per_ex=1, ignore_z=False, latents_seed=None):
        """Returns representations for [x].

        Args:
        x   -- BSxCxHxW images to compute representations for
        z   -- list of noises, one for each IP block
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx,block in enumerate(self.blocks):
            if idx in self.block_idx2latent_idx:
                x = block(x,
                    latent_codes=None if latent_codes is None else latent_codes[:, self.block_idx2latent_idx[idx]],
                    codes_per_ex=codes_per_ex if idx == self.first_ip_idx else 1,
                    seed=latents_seed,
                    ignore_z=ignore_z)
            else:
                x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            return self.fc_norm(x)
        else:
            x = self.norm(x)
            return x[:, 0]

    def forward(self, x, **kwargs):
        """Forward function for getting class predictions.

        Requires that set_latent_spec() has been called on the model.

        Args:
        x       -- BSxCxHxW tensor of images
        z       -- filled-in latent specification of size matching tha tof [x]
        z_per_x -- number of latents for example if [z] is None
        """
        return self.forward_head(self.forward_features(x, **kwargs))

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
        blocks = []
        for idx,b in enumerate(self.blocks):
            prev_block_idx = idx - 1
            if (prev_block_idx in self.idx2ip_method
                and isinstance(self.idx2ip_method[prev_block_idx], LocalAdaIN)):
                blocks.append(Affine.make_block_start_with_affine(b))
            else:
                blocks.append(b)

        # Make some blocks implicit probabilistic given the IP specification
        self.blocks = nn.ModuleList([IPBlock(b, ipm) if ipm else b for b,ipm in zip(blocks, self.idx2ip_method.values())])

        # Create a dictionary that gives the index to the latent codes for the
        # index of each index of the IP blocks
        counter = 0
        self.block_idx2latent_idx = {}
        for idx,ipm in self.idx2ip_method.items():
            if ipm:
                self.block_idx2latent_idx[idx] = counter
                counter += 1
        
        self.first_ip_idx = min(self.block_idx2latent_idx.keys())

        # If the last encoder block ends with a LocalAdaIN, the decoder norm and
        # the first norm of the first decoder block need to become Affine layers
        if isinstance(self.idx2ip_method[len(self.blocks) - 1], LocalAdaIN):
            self.norm = Affine.from_layernorm(self.norm)
            self.decoder_blocks[0] = Affine.make_block_start_with_affine(self.decoder_blocks[0])
        ########################################################################

        tqdm.write(f"{self.__class__.__name__} [num_blocks {len(self.blocks)} | num_params {sum(p.numel() for p in self.parameters() if p.requires_grad)}]")

    def get_latent_codes(self, bs, **kwargs):
        """Returns [bs] latent codes to be input to the model."""
        result = [block.get_latent_codes(bs, **kwargs)
            for idx,block in enumerate(self.blocks)
            if idx in self.block_idx2latent_idx]
        return torch.stack(result, dim=1)

    def forward_encoder(self, x, mask_ratio=0.75, mask_seed=None,
        mask_codes=None, latents_seed=None, latent_codes=None, codes_per_ex=1,
        ignore_z=False):
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
        x, mask, ids_restore = self.random_masking(x,
            mask_ratio=mask_ratio,
            mask_codes=mask_codes,
            mask_seed=mask_seed)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for idx,block in enumerate(self.blocks):
            if idx in self.block_idx2latent_idx:
                x = block(x,
                    latent_codes=None if latent_codes is None else latent_codes[:, self.block_idx2latent_idx[idx]],
                    codes_per_ex=codes_per_ex if idx == self.first_ip_idx else 1,
                    seed=latents_seed,
                    ignore_z=ignore_z)
            else:
                x = block(x)

        x = self.norm(x)
        return x, mask, ids_restore

    def forward(self, x, mask_ratio=0.75, reduction="mean", return_all=False,
        mask_seed=None, mask_codes=None, latents_seed=None, latent_codes=None,
        codes_per_ex=1, ignore_z=False):
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
        latent, mask, ids_restore = self.forward_encoder(x,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            mask_codes=mask_codes,
            latents_seed=latents_seed,
            latent_codes=latent_codes,
            codes_per_ex=codes_per_ex,
            ignore_z=ignore_z)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask,
            reduction=reduction,
            mask_ratio=mask_ratio)
        
        if return_all:
            pred, mask = restore_model_outputs(pred, mask, self.patch_embed,
                self.unpatchify)
            return loss, pred, mask
        else:
            return loss

class MaskedIPViTSampledMaskToken(MaskedAutoencoderViT):
    """Masked VAE with VisionTransformer backbone.

    Args:
    mae_model       -- None or MaskedAutoencoderViT instance. If the former,
                        values in [kwargs] are used to specify the ViT
                        architecture; if the latter, the architecture of
                        [mae] model is used to for this one
    **kwargs        -- same kwargs as for a MaskedAutoencoderViT. Ignored if
                        [mae_model] is specified.
    """
    def __init__(self, mae_model=None, **kwargs):
        if mae_model is None:
            super(MaskedIPViTSampledMaskToken, self).__init__(**kwargs)
        else:
            super(MaskedIPViTSampledMaskToken, self).__init__(**mae_model.kwargs)
            self.load_state_dict(mae_model.state_dict(), strict=False)

        self.mask_token_shape = self.patch_embed.patch_size[0] ** 2 * 2
        self.adain = AdaIN(c=self.mask_token_shape)

        tqdm.write(f"{self.__class__.__name__} [num_blocks {len(self.blocks)} | num_params {sum(p.numel() for p in self.parameters() if p.requires_grad)}]")

    def get_latent_codes(self, bs=1, device=device, seed=None):
        x = torch.zeros(bs, self.adain.code_dim, device=device)
        g = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        x.normal_(generator=g)
        return x

    def forward_decoder(self, x, ids_restore, latent_codes=None, codes_per_ex=1, latents_seed=None, ignore_z=False):
        if latent_codes is None:
            latent_codes = self.get_latent_codes(bs=len(x) * codes_per_ex, device=x.device, seed=seed)

        x = x.repeat_interleave(len(latent_codes) // len(x), dim=0)
        ids_restore = ids_restore.repeat_interleave(len(x) // len(ids_restore), dim=0)
        mask_tokens = self.adain(self.mask_token,
            latent_codes=latent_codes,
            latents_seed=latents_seed,
            ignore_z=ignore_z)
        mask_tokens = mask_tokens.repeat(1, ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = self.decoder_embed(x)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward(self, x, mask_codes=None, latent_codes=None, mask_ratio=0.75,
        reduction="mean", return_all=False, mask_seed=None, latents_seed=None, codes_per_ex=1):
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
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, mask_codes=mask_codes, mask_seed=mask_seed)
        pred = self.forward_decoder(latent, ids_restore,
            latent_codes=latent_codes,
            latents_seed=latents_seed,
            codes_per_ex=codes_per_ex)
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

def get_masked_ipvit_model(args):
    """Returns an MaskedIPViT model with the weights and architecture of an MAE
    model specified in [args. Resuming a MaskedIPViT model is not handeled, ie.
    the model weights are pure MAE weights.
    """
    # Instantiate the MAE model we're finetuning
    if args.arch == "vit_base":
        mae_model_state = torch.load(f"{os.path.dirname(__file__)}/mae_checkpoints/mae_pretrain_vit_base_full.pth")["model"]
        mae = mae_vit_base_patch16(norm_pix_loss=args.norm_pix_loss)
    elif args.arch == "vit_base_vis":
        mae_model_state = torch.load(f"{os.path.dirname(__file__)}/mae_checkpoints/mae_visualize_vit_base.pth")["model"]
        mae = mae_vit_base_patch16(norm_pix_loss=False)
    elif args.arch == "vit_large":
        mae_model_state = torch.load(f"{os.path.dirname(__file__)}/mae_checkpoints/mae_pretrain_vit_large_full.pth")["model"]
        mae = mae_vit_large_patch16(norm_pix_loss=args.norm_pix_loss)
    else:
        raise NotImplementedError()
    mae.load_state_dict(mae_model_state)
    model_kwargs = {"mae_model": mae} if args.finetune else mae.kwargs

    model = MaskedIPViT(parse_ip_spec(args), **model_kwargs)
    
    model = model.to(torch.float32)
    return model

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

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
