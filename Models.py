"""

NOTES:
1. Pretrained models are available at https://github.com/facebookresearch/mae/issues/8, which isn't the same as those directly linked in the repo.
"""

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block

from original_code.util import pos_embed as PE

from Utils import *

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024,
        depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8,
        decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
        norm_pix_loss=False, verbose=False):

        super(MaskedAutoencoderViT, self).__init__()

        # Save variables needed in making this variational
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


        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        if verbose:
            tqdm.write(f"LOG: Constructed MaskedAutoencoderViT model | num_blocks {len(self.blocks)} | num_params {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def initialize_weights(self):
        # initialization
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
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

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
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
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

    def forward_loss(self, imgs, pred, mask, reduction="mean"):
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

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss * mask

        if reduction == "mean":
            return (loss.sum() / mask.sum()).unsqueeze(0)  # The 0-1ness of [mask] makes this not dependent on batch size
        elif reduction == "batch":
            return torch.mean(loss, axis=1)
        else:
            raise NotImplementedError()

    def forward(self, imgs, mask_ratio=0.75, reduction="mean"):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, reduction=reduction)
        return loss, pred, mask

def extend_idx2v_method(idx2v_method, length):
    """
    """
    return {k: False for k in range(length)} | idx2v_method

class VariationalBlock(nn.Module):
    """Wraps a vision transformer block and adds noise to its output.
    Args:
    block       -- the vision transformer block to wrap
    v_method    -- how to add noise (a) 'add' to have the noise added to the
                    block's output, or (b) an nn.Module taking the output and
                    the noise as input and producing a new output with the same
                    dimenion as the block's output
    """
    def __init__(self, block, v_method="add"):
        super(VariationalBlock, self).__init__()
        self.block = block
        self.v_method = v_method

    def get_latent_shape(self, test_input=torch.ones(4, 3, 224, 224, device=device)):
        """Returns the latent shape minus the batch dimension for the network.

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        """
        block_output = self.block(test_input)
        if self.v_method == "add":
            return block_output.shape[1:]
        elif isinstance(self.v_method, nn.Module):
            return self.v_method.get_latent_shape(block_output)
        else:
            raise NotImplementedError()

    def forward(self, x, z):
        """Forward function.

        Args:
        x   -- input to the wrapped block
        z   -- either a tensor to use as noise or an nn.Module whose forward
                function returns the noise. One might use the former type for
                IMLE training and the latter for density-based linear probing
        """
        fx = self.block(x)
        z = z() if isinstance(z, nn.Module) else z

        if self.v_method == "add":
            fx = torch.repeat_interleave(fx, z.shape[0] // fx.shape[0], dim=0)
            return fx + z
        elif isinstance(self.v_method, nn.Module):
            return self.v_method(fx, z)
        else:
            raise NotImplementedError()

class VariationaViT(timm.models.vision_transformer.VisionTransformer):
    """ViT, but with the ability to add Gaussian noise at some place in the
    forward pass.

    Args:
    idx2v_method    -- mapping from from indices to the blocks to whether and
                        how they should be variational. If the ith value in the
                        mapping is False, the ith block will not be made
                        variational. Otherwise, it will be replaced with a
                        VariationalBlock using the value as [v_method]
    kwargs          -- arguments for constructing the ViT architecture
    """

    def __init__(self, idx2v_modules, v_mae_model=None, **kwargs):
        if v_mae_model is None:
            super(VisionTransformer, self).__init__(**kwargs)
        else:
            super(VisionTransformer, self).__init__(**v_mae_model.kwargs)

        # Replace the normal ModuleList [blocks] with a dictionary mapping from
        # block indices to the blocks, with some of the blocks made variational
        # (ie. accept a representation and noise as input, and fuse the two in
        # the output). This is way of adding variationalness can be adapted to
        # many different architectures.
        self.idx2v_method = extend_idx2v_method(idx2v_method, len(self.blocks))
        idx2block = [VariationalBlock(b, vm) if vm else b
            for b,vm in zip(self.blocks, self.idx2v_method.values())]
        self.idx2block = {str(idx): b for idx,b in enumerate(idx2block)}
        self.idx2block = nn.ModuleDict(self.idx2block)
        del self.blocks

        self.block_idx2z_idx = {b_idx: z_idx
            for z_idx,b_idx in enumerate([
                b_idx for b_idx,vm in self.idx2v_method.items() if vm])}

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        if if mae_model is None:
            self.load_state_dict(v_mae_model.state_dict())

    def get_latent_shape(self, test_input=torch.ones(4, 3, 224, 224, device=device), mask_ratio=1):
        """Returns the latent shape minus the batch dimension for the network.

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        mask_ratio  -- mask ratio. Ignored.
        """
        shapes = []
        with torch.no_grad():
            # Forward method, but can adds computed shapes to [shapes]
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
                    s = block.get_latent_shape(test_input=x)
                    shapes.append(s)
                    x = block(x, torch.ones(len(x), *s, device=device))
                else:
                    x = block(x)

        if len(set(shapes)) == 1:
            return {"mask_noise": {"shape": mask_noise_shape, "batch_dim": 0},
                "latents": {"shape": (len(shapes),) + shapes[0], "batch_dim": 0}}
        else:
            raise ValueError(f"Got multiple shapes for latent codes: {shapes}")

    def forward_features(self, x, z):
        """Returns representations for [x].

        Args:
        x   -- BSxCxHxW images to compute representations for
        z   -- list of noises, one for each variational block
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # The only change from a normal ViT architecture. When we encounter a
        # VariationalBlock, it gets to have its noise fed to it too.
        for idx,block in self.blocks.items():
            if int(idx) in self.block_idx2z_idx:
                x = block(x, z[self.block_idx2z_idx[idx]])
            else:
                x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, z):
        fx = self.forward_features(x, z)
        fx = self.forward_head(fx)
        return fx

class MaskedViTVAE(MaskedAutoencoderViT):
    """Masked VAE with VisionTransformer backbone.

    Args:
    idx2v_method    --  mapping from from indices to the blocks to whether and
                        how they should be variational. If the ith value in the
                        mapping is False, the ith block will not be made
                        variational. Otherwise, it will be replaced with a
                        VariationalBlock using the value as [v_method]
    mae_model       -- None or MaskedAutoencoderViT instance. If the former,
                        values in [kwargs] are used to specify the ViT
                        architecture; if the latter, the architecture of
                        [mae] model is used to for this one
    **kwargs        -- same kwargs as for a MaskedAutoencoderViT. Ignored if
                        [mae_model] is specified.
    """
    def __init__(self, idx2v_method, mae_model=None, **kwargs):
        if mae_model is None:
            super(MaskedViTVAE, self).__init__(**kwargs)
        else:
            super(MaskedViTVAE, self).__init__(**mae_model.kwargs)
            self.load_state_dict(mae_model.state_dict())

        # Replace the normal ModuleList [blocks] with a dictionary mapping from
        # block indices to the blocks, with some of the blocks made variational
        # (ie. accept a representation and noise as input, and fuse the two in
        # the output). This is way of adding variationalness can be adapted to
        # many different architectures.
        self.idx2v_method = extend_idx2v_method(idx2v_method, len(self.blocks))
        idx2block = [VariationalBlock(b, vm) if vm else b
            for b,vm in zip(self.blocks, self.idx2v_method.values())]
        self.idx2block = {str(idx): b for idx,b in enumerate(idx2block)}
        self.idx2block = nn.ModuleDict(self.idx2block)
        del self.blocks

        self.block_idx2z_idx = {b_idx: z_idx
            for z_idx,b_idx in enumerate([
                b_idx for b_idx,vm in self.idx2v_method.items() if vm])}

        tqdm.write(f"LOG: Constructed MaskedViTVAE model | num_blocks {len(self.idx2block)} | latent to block mapping {self.block_idx2z_idx} | num_params {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

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
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        mask_noise = torch.rand(N, L, device=x.device) if mask_noise is None else mask_noise

        # sort noise for each sample
        ids_shuffle = torch.argsort(mask_noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def get_latent_shape(self, test_input=torch.ones(4, 3, 224, 224, device=device),
        mask_ratio=.75):
        """Returns the latent shape minus the batch dimension for the network.
        Concretely, this is a list of tuples. Th

        Args:
        test_input  -- a BSxCxHxW tensor to be used as the test input
        mask_ratio  -- mask ratio
        """
        shapes = []
        with torch.no_grad():
            # Forward method, but can adds computed shapes to [shapes]
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
                    s = block.get_latent_shape(test_input=x)
                    shapes.append(s)
                    x = block(x, torch.ones(len(x), *s, device=device))
                else:
                    x = block(x)

        if len(set(shapes)) == 1:
            return {"mask_noise": {"shape": mask_noise_shape, "batch_dim": 0},
                "latents": {"shape": (len(shapes),) + shapes[0], "batch_dim": 0}}
        else:
            raise ValueError(f"Got multiple shapes for latent codes: {shapes}")

    def forward_encoder(self, x, z, mask_ratio):
        """Forward function for the encoder.

        Args:
        x           -- BSxCxHxW tensor of images to encode
        z           -- dictionary with keys 'mask_noise' and 'latents', which
                        should each map to a tensor filled with noise with
                        shapes given by the output of get_latent_shape()
        mask_ratio  -- mask ratio
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio,
            mask_noise=z["mask_noise"])

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # The only change from a normal ViT architecture. When we encounter a
        # VariationalBlock, it gets to have its noise fed to it too.
        for idx,block in self.idx2block.items():
            if int(idx) in self.block_idx2z_idx:
                x = block(x, z["latents"][:, self.block_idx2z_idx[int(idx)]])
            else:
                x = block(x)

        x = self.norm(x)
        return x, mask, ids_restore

    def forward(self, x, z, mask_ratio=0.75, reduction="mean"):
        """
        Args:
        x           -- BSxCxHxW tensor of images to encode
        z           -- dictionary with keys 'mask_noise' and 'latents', which
                        should each map to a tensor filled with noise with
                        shapes given by the output of get_latent_shape()
        mask_ratio  -- mask ratio
        """
        latent, mask, ids_restore = self.forward_encoder(x, z, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask, reduction=reduction)
        return loss, pred, mask


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
