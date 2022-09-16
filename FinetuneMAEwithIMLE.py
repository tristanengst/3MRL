import argparse
from itertools import chain
from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torch
from torch.optim import AdamW
import torch.nn
from torch.utils.data import DataLoader, Subset, Dataset

from ApexUtils import *
from Data import *
from Models import *
from Utils import *

def model_folder(args):
    """Returns the folder to which to save a model built with [args]."""
    uid = args.uid if args.job_id is None else args.job_id
    data = data_without_split_or_path(args.data_tr)
    folder = f"{project_dir}/models/{data}-bs{args.bs}-epochs{args.epochs}-ipe{args.ipe}-lr{args.lr:.2e}-ns{tuple_to_str(args.ns)}-{uid}{suffix_str(args)}"

    conditional_safe_make_directory(folder)
    if not os.path.exists(f"{folder}/config.json"):
        with open(f"{folder}/config.json", "w+") as f:
            json.dump(vars(args), f)
    return folder

class ImageLatentDataset(Dataset):
    """Dataset for loading images and latents to a MaskedViTVAE model. The
    provided collate function must be used in any DataLoaders wrappeding this
    dataset.

    Args:
    images      -- NxCxHxW tensor of images
    mask_noises -- NxD tensor of mask noises
    latents     -- Nx... tensor of latents for the model
    """
    def __init__(self, images, mask_noises, latents):
        super(ImageLatentDataset, self).__init__()
        self.images = images
        self.mask_noises = mask_noises
        self.latents = latents

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.mask_noises[idx], self.latents[idx]

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        mask_noises = torch.stack([b[1] for b in batch])
        latents = torch.stack([b[2] for b in batch])
        return images, {"mask_noise": mask_noises, "latents": latents}

def get_image_latent_dataset(model, dataset, latent_spec, args):
    """Returns an ImageLatent dataset constructed using the data in [dataset].
    This dataset can be used for one epoch's worth of training.

    Args:
    model       -- model to get codes for
    dataset     -- dataset containing the data to sample
    latent_spec -- dictionary of latent code shapes
    args        -- Namespace with --sp, --ns, and --code_bs arguments
    """
    with torch.no_grad():
        least_losses = torch.ones(len(dataset), device=device) * float("inf")
        initial_codes = sample_latent_dict(latent_spec, len(dataset),
            device="cpu",
            noise="ones")
        best_latents = initial_codes["latents"]
        mask_noise = initial_codes["mask_noise"]
        latents_only_spec = {"latents": latent_spec["latents"]}

        loader = DataLoader(dataset,
            batch_size=args.code_bs,
            pin_memory=True,
            num_workers=12,
            drop_last=False)

        all_images = []
        for idx,(x,_) in tqdm(enumerate(loader),
            desc="SAMPLING: Chunks of batch",
            total=len(loader),
            leave=False,
            dynamic_ncols=True):

            all_images.append(x)
            bs = len(x)
            start = idx * args.code_bs
            stop = min(len(dataset), (idx + 1) * args.code_bs)
            mask_noise_ = mask_noise[start:stop]
            mask_noise_ = mask_noise_.repeat_interleave(args.sp, dim=0)

            for idx in tqdm(range(args.ns // args.sp),
                desc=f"SAMPLING code_bs {args.code_bs} | sp {args.sp}: Iterations over code batch size",
                leave=False,
                dynamic_ncols=True):

                latents = sample_latent_dict(latents_only_spec, bs * args.sp)
                z = {"mask_noise": mask_noise_} | latents
                losses, _, _ = model(x_sampling, z,
                    mask_ratio=args.mask_ratio,
                    reduction="batch")
                _, idxs = torch.min(losses.view(bs, args.sp), dim=1)

                new_codes = z["latents"]
                new_codes = new_codes.view((bs, args.sp) + new_codes.shape[1:])
                new_codes = new_codes[torch.arange(bs), idxs]
                losses = losses.view(bs, args.sp)[torch.arange(bs), idxs]

                change_idxs = losses < least_losses[start:stop]
                best_latents[start:stop][change_idxs] = new_codes[change_idxs]
                least_losses[start:stop][change_idxs] = losses[change_idxs]

    all_images = torch.cat(all_images, axis=0).cpu()
    return ImageLatentDataset(all_images, mask_noise.cpu(), best_latents.cpu())

def validate(model, data_fn, data_te, args):
    pass

def parse_v_spec(args):
    """Returns the variational specification specifed in [args] but mapped to
    more useful values.
    """
    def parse_v_spec_helper(s):
        if s == "add" or not s:
            return s
        elif s.starswith("downsample_mlp_"):
            hidden_dim = int(s[len("downsample_mlp_"):])
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    key_args = [int(k) for idx,k in enumerate(args.v_spec) if idx % 2 == 0]
    val_args = [v for idx,v in enumerate(args.v_spec) if idx % 2 == 1]
    spec = {k: v for k,v in zip(key_args, val_args)}

    max_specified_key = max(key_args)
    non_variational_layers = {idx: False for idx in range(max_specified_key)}
    spec = non_variational_layers | spec

    return {k: parse_v_spec_helper(v) for k,v in spec.items()}

def get_args(args=None):
    P = argparse.ArgumentParser()
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="Type of W&B logging")


    P.add_argument("--arch", choices=["base_pretrained"], default="base_pretrained",
        help="Type of ViT model to use")
    P.add_argument("--resume", default=None,
        help="Path to checkpoint to resume")
    P.add_argument("--data_tr", required=True, choices=get_available_datasets(),
        help="String specifying training data")
    P.add_argument("--data_fn", required=True, choices=get_available_datasets() + ["cv"],
        help="String specifying finetuning data")
    P.add_argument("--data_te", required=True, choices=get_available_datasets() + ["cv"],
        help="String specifying testing data")
    P.add_argument("--data_path", default=data_dir,
        help="Path to where datasets are stored")

    P.add_argument("--finetune", choices=[0, 1], type=int, default=1,
        help="Whether to finetune an existing MAE model or train from scratch")
    P.add_argument("--v_spec", nargs="+",
        help="Specification for making the autoencoder variational")

    P.add_argument("--epochs", type=int, default=64,
        help="Number of sampling/training steps to run")
    P.add_argument("--ex_per_epoch", type=int, default=512,
        help="Number of examples to use in each sampling")
    P.add_argument("--code_bs", type=int, default=4,
        help="Batch size for sampling")
    P.add_argument("--ns", type=int, default=1024,
        help="Number of latent codes to generate per image during sampling")
    P.add_argument("--sp", type=int, default=4,
        help="Per-image latent code parallelism during sampling")
    P.add_argument("--mini_bs", type=int, default=64,
        help="Batch size for training")
    P.add_argument("--ipe", type=int, default=10240,
        help="Gradient steps per epoch (use a multiple of --bs // --mini_bs)")

    P.add_argument("--lr", type=float, default=1e-3,
        help="Base learning rate")

    P.add_argument("--mask_ratio", type=float, default=.75,
        help="Mask ratio for the model")
    P.add_argument("--res", default=256, type=int,
        help="Resolution to get data at")
    P.add_argument("--input_size", default=224, type=int,
        help="Size of (cropped) images to feed to the model")
    P.add_argument("--gpus", nargs="+", default=[0, 1], type=int,
        help="Device IDs of GPUs to use")

    P.add_argument("--job_id", type=int, default=None,
        help="SLURM job_id if available")

    args = P.parse_args() if args is None else P.parse_args(args)

    args.uid = wandb.util.generate_id() if args.job_id is None else args.job_id
    return args

if __name__ == "__main__":
    args = get_args()

    ############################################################################
    # Load resumed things or instantiate them
    ############################################################################
    if args.resume is None:
        wandb.init(anonymous="allow", id=args.uid, config=args,
            mode=args.wandb, project="3MRD",
            name=os.path.basename(model_folder(args)))

        if args.arch == "base_pretrained":
            mae_model_state = torch.load(f"{project_dir}/mae_checkpoints/mae_pretrain_vit_base_full.pth")["model"]
            mae = mae_vit_base_patch16()
            mae.load_state_dict(mae_model_state)
        else:
            raise NotImplementedError()

        model_kwargs = {"mae_model": mae} if args.finetune else mae.kwargs
        model = MaskedViTVAE(parse_v_spec(args), **model_kwargs)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
        last_epoch = -1
    else:
        raise NotImplementedError()


    ############################################################################
    # Get the datasets
    ############################################################################
    data_tr, data_fn, data_te = get_imagefolder_data(args.data_tr, args.data_fn, args.data_te,
        res=args.res,
        data_path=args.data_path)

    data_tr = XYDataset(data_tr, transform=get_train_transforms(args),
        normalize=False) # Set to True when not debugging)

    # Construct [latent_shapes_dict] a dictionary giving all the latent shapes
    # needed to get noise to run [model]. We can pass it along with a [bs]
    # argument to sample_latent_dict() and it will give us the latents we need.
    t = torch.stack([data_tr[idx][0] for idx in range(args.sp * args.code_bs)])
    latent_spec = model.module.get_latent_shape(t.to(device))

    ############################################################################
    # Complete training setup
    ############################################################################
    kkm = KOrKMinusOne(range(len(data_tr)), shuffle=True)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.epochs * args.ipe,
        max_lr=args.lr,
        min_lr=args.lr / 100,
        last_epoch=-1 if last_epoch == -1 else last_epoch * args.ipe)

    ############################################################################
    # Begin training
    ############################################################################

    log_iter = int(args.epochs * args.ipe / 10000)
    cur_step = (last_epoch + 1) * args.ipe
    for epoch in tqdm(range(args.epochs),
        desc="TRAINING: Epochs",
        dynamic_ncols=True,
        leave=False):

        batch_data_tr = Subset(data_tr, indices=kkm.pop_k(args.ex_per_epoch))
        batch_dataset = get_image_latent_dataset(model, batch_data_tr,
            latent_spec, args)
        batch_loader = DataLoader(batch_dataset,
            batch_size=args.mini_bs,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=ImageLatentDataset.collate_fn)
        num_passes_over_loader = max(1, args.ipe // len(batch_loader))
        batch_loader = chain(*[batch_loader] * num_passes_over_loader)

        for x,z in tqdm(batch_loader,
            desc="TRAINING: Minibatches",
            dynamic_ncols=True,
            leave=False):

            with torch.cuda.amp.autocast():
                loss, _, _ = model(x, z, mask_ratio=args.mask_ratio)
                loss = torch.mean(loss)

            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            cur_step += 1

            if cur_step % log_iter == 0:
                wandb.log({"loss/pretrain": loss.item(),
                    "lr": scheduler.get_lr()},
                    step=cur_step)

        ########################################################################
        # Validate and save a checkpoint
        ########################################################################
        if epoch % args.eval_iter == 0:
            model = model.cpu()
            val_model = VariationaViT(parse_v_spec(args), v_mae_model=model)
            val_model = nn.DataParallel(val_model, device_idxs=args.gpus).to(device)

            val_results = linear_probe_one_aug(val_model,
                data_fn=data_fn,
                data_te=data_te,
                latent_spec={"latents": latent_spec["latents"]},
                args=args)

            wandb.log(val_results)
