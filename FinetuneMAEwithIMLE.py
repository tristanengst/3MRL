import argparse
from itertools import chain
from tqdm import tqdm
import wandb

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torch
from torch.optim import AdamW
import torch.nn
from torch.utils.data import DataLoader, Subset, Dataset

from original_code.models_mae import mae_vit_base_patch16 as foo

from Augmentation import *
from ApexUtils import *
from Data import *
from LinearProbe import linear_probe
from Models import *
from Utils import *

def model_folder(args):
    """Returns the folder to which to save a model built with [args]."""
    uid = args.uid if args.job_id is None else args.job_id
    data = data_without_split_or_path(args.data_tr)
    folder = f"{project_dir}/models/{data}-bs{args.ex_per_epoch}-epochs{args.epochs}-ipe{args.ipe}-lr{args.lr:.2e}-ns{tuple_to_str(args.ns)}-{uid}{suffix_str(args)}"

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
            num_workers=args.num_workers,
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

            for idx in tqdm(range(args.ns // args.sp),
                desc=f"SAMPLING code_bs {args.code_bs} | sp {args.sp}: Iterations over code batch size",
                leave=False,
                dynamic_ncols=True):

                latents = sample_latent_dict(latents_only_spec, bs * args.sp)
                z = {"mask_noise": mask_noise_} | latents
                with torch.cuda.amp.autocast():
                    losses = model(x, z, mask_ratio=args.mask_ratio,
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

def validate(model, data_tr, data_fn, data_te, args):
    """Returns a dictionary of validation data about [model].

    Args:
    model       -- MaskedVAEViT model
    data_tr     -- Dataset of training data for reconstruction
    data_tr     -- Dataset of data for linear probe training
    data_tr     -- Dataset of testing data for linear probes and reconstruction
    args        -- Namespace with relevant parameters
    """
    def get_reconstruction_images_loss(model, dataset, latent_spec, args):
        """Returns an (loss, image_grid) tuple, where [loss] is the average loss
        of [model] on reconstructing images from [dataset] and [image_grid]  is
        a grid of images for qualitatively evaluating the reconstructions.

        Args:
        model       -- MaskedVAEViT model
        dataset     -- Dataset containing all the data to use in reconstruction
        latent_spec -- dictionary giving the latent specification for [model]
        args        -- Namespace with relevant parameters
        """
        loader = DataLoader(dataset,
            batch_size=args.code_bs,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)

        images, preds, masks, total_loss = [], [], [], 0
        with torch.no_grad():
            for x,_ in tqdm(loader,
                desc="Validation: computing reconstruction loss and image grid",
                leave=False,
                dynamic_ncols=True):

                z_bs = {"mask_noise": len(x), "latents": len(x) * args.z_per_ex}
                z = sample_latent_dict(latent_spec, z_bs, noise=args.noise)
                with torch.cuda.amp.autocast():
                    loss, pred, mask = model(x, z,
                        mask_ratio=args.mask_ratio,
                        return_all=True)
                total_loss += (loss.mean() * len(x)).item()
                images.append(de_normalize(x).cpu())
                preds.append(pred.cpu())
                masks.append(mask.cpu())

        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)
        preds = torch.cat(preds, dim=0)
        preds = preds.view(len(dataset), args.z_per_ex, *preds.shape[1:])
        image_grid = [[image] + [p for p in pred]
            for image,pred in zip(images, preds)]

        return total_loss / (len(dataset) * args.z_per_ex), image_grid

    ############################################################################
    # See how well the model is doing as a VAE
    ############################################################################
    t = torch.stack([data_tr[0][0]])
    latent_spec = model.module.get_latent_spec(t.to(device),
        mask_ratio=args.mask_ratio)
    idxs_tr = torch.linspace(0, len(data_tr) - 1, args.num_ex_for_eval_tr)
    idxs_te = torch.linspace(0, len(data_te) - 1, args.num_ex_for_eval_te)
    vae_loss_tr, images_tr = get_reconstruction_images_loss(model,
        Subset(data_tr, [int(idx) for idx in idxs_tr.tolist()]),
        latent_spec, args)
    vae_loss_te, images_te = get_reconstruction_images_loss(model,
        Subset(data_te, [int(idx) for idx in idxs_te.tolist()]),
        latent_spec, args)

    ############################################################################
    # Do linear probing to see how well the model's weights work in ViT
    ############################################################################
    lin_probe_results = linear_probe(model, data_fn=data_fn, data_te=data_te,
        args=args)
    model = model.to(device)

    return lin_probe_results | {"loss/vae_test": vae_loss_te,
        "images/vae_train": images_to_pil_image(images_tr),
        "images/vae_test": images_to_pil_image(images_te)}

def get_args(args=None):
    P = argparse.ArgumentParser()
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="Type of W&B logging")
    P.add_argument("--finetune", choices=[0, 1], type=int, default=1,
        help="Whether to finetune an existing MAE model or train from scratch")
    P.add_argument("--v_spec", nargs="*", default=[],
        help="Specification for making the autoencoder variational")
    P.add_argument("--arch", choices=["base_pretrained"], default="base_pretrained",
        help="Type of ViT model to use")
    P.add_argument("--resume", default=None,
        help="Path to checkpoint to resume")
    P.add_argument("--suffix", type=str, default=None,
        help="Optional suffix")

    # Data arguments
    P.add_argument("--data_tr", required=True, choices=get_available_datasets(),
        help="String specifying training data")
    P.add_argument("--data_fn", required=True, choices=get_available_datasets() + ["cv"],
        help="String specifying finetuning data")
    P.add_argument("--data_te", required=True, choices=get_available_datasets() + ["cv"],
        help="String specifying testing data")
    P.add_argument("--data_path", default=data_dir,
        help="Path to where datasets are stored")

    # Evaluation arguments. Optimization here isn't too important because we can
    # use saved models to get representations and do hyperparameter tuning.
    P.add_argument("--num_ex_for_eval_tr", default=8,
        help="Number of training examples for logging")
    P.add_argument("--num_ex_for_eval_te", default=8,
        help="Number of training examples for logging")
    P.add_argument("--z_per_ex", default=6,
        help="Number of latents per example for logging")
    P.add_argument("--z_per_ex_eval", type=int, default=16,
        help="Number of codes to use per example in linear probing")
    P.add_argument("--trials", type=int, default=1,
        help="Number of trials to use in linear probe")
    P.add_argument("--num_val_ex", type=int, default=1024,
        help="Number of examples to use for validation")
    P.add_argument("--val_epochs", type=int, default=120,
        help="Number of epochs to use in linear probing")
    P.add_argument("--val_bs", type=int, default=8,
        help="Batch size to use in supervised linear probing")
    P.add_argument("--val_lr", type=float, default=1e-3,
        help="Learning rate to use in linear probing")
    P.add_argument("--global_pool", type=int, default=0, choices=[0, 1],
        help="Whether to use global pooling in forming representations")

    # Training arguments
    P.add_argument("--epochs", type=int, default=64,
        help="Number of sampling/training steps to run")
    P.add_argument("--ex_per_epoch", type=int, default=512,
        help="Number of examples to use in each sampling")
    P.add_argument("--code_bs", type=int, default=4,
        help="Batch size for sampling")
    P.add_argument("--ns", type=int, default=1024,
        help="Number of latents from which to choose per image for sampling")
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
    P.add_argument("--n_ramp", type=int, default=10,
        help="Number of epochs of linear learning rate warmup")
    P.add_argument("--res", default=256, type=int,
        help="Resolution to get data at")
    P.add_argument("--input_size", default=224, type=int,
        help="Size of (cropped) images to feed to the model")
    P.add_argument("--noise", default="gaussian", choices=["gaussian", "zeros"],
        help="Kind of noise to add inside the model")

    # Hardware arguments
    P.add_argument("--gpus", nargs="+", default=[0, 1], type=int,
        help="Device IDs of GPUs to use")
    P.add_argument("--job_id", type=int, default=None,
        help="SLURM job_id if available")
    P.add_argument("--num_workers", type=int, default=20,
        help="Number of DataLoader workers")

    args = P.parse_args() if args is None else P.parse_args(args)

    args.uid = wandb.util.generate_id() if args.job_id is None else args.job_id

    if len(args.v_spec) == 0:
        tqdm.write(f"WARNING: empty --v_spec precludes model from returning multiple outputs for one input. Consider adding a variational block with --noise set to 'zeros'")
        args.z_per_ex = 1
        args.z_per_ex_eval = 1
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
        model = MaskedVAEViT(parse_variational_spec(args), **model_kwargs)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
        last_epoch = -1
    else:
        raise NotImplementedError()

    save_dir = model_folder(args)
    ############################################################################
    # Get the datasets
    ############################################################################
    data_tr, data_fn, data_te = get_imagefolder_data(args.data_tr, args.data_fn,
        args.data_te, res=args.res, data_path=args.data_path)
    data_tr = XYDataset(data_tr, transform=get_train_transforms(args))
    data_fn = XYDataset(data_fn, transform=get_finetuning_transforms(args))
    data_te = XYDataset(data_te, transform=get_test_transforms(args))






    # Construct [latent_shapes_dict] a dictionary giving all the latent shapes
    # needed to get noise to run [model]. We can pass it along with a [bs]
    # argument to sample_latent_dict() and it will give us the latents we need.
    t = torch.stack([data_tr[idx][0] for idx in range(args.sp * args.code_bs)])
    latent_spec = model.module.get_latent_spec(t.to(device),
        mask_ratio=args.mask_ratio)

    ############################################################################
    # Complete training setup
    ############################################################################
    kkm = KOrKMinusOne(range(len(data_tr)), shuffle=True)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.epochs * args.ipe,
        warmup_steps=args.n_ramp * args.ipe,
        max_lr=args.lr,
        min_lr=args.lr / 100,
        last_epoch=-1 if last_epoch == -1 else last_epoch * args.ipe)

    ############################################################################
    # Begin training
    ############################################################################
    if last_epoch == -1:
        results = validate(model, data_tr, data_fn, data_te, args)
        if not os.path.exists(f"{save_dir}/images"):
            os.makedirs(f"{save_dir}/images")
        results["images/vae_train"].save(f"{save_dir}/images/0_train.png")
        results["images/vae_test"].save(f"{save_dir}/images/0_test.png")
        results["images/vae_train"] = wandb.Image(results["images/vae_train"])
        results["images/vae_test"] = wandb.Image(results["images/vae_test"])
        wandb.log(results, step=0)

        tqdm.write(f"Epoch {0:4}/{args.epochs} | \
            loss/vae_test {results['loss/vae_test']} | \
            loss/lin_probe_train {results['loss/lin_probe_train']} | \
            loss/lin_probe_test {results['loss/lin_probe_test']} | \
            accuracy/lin_probe_test {results['accuracy/lin_probe_test']:.3f}±{results['accuracy_lin_probe_test_conf']:.3f}")

    scaler = torch.cuda.amp.GradScaler()
    log_iter = int(args.epochs * args.ipe / 10000)
    cur_step = 0 if last_epoch == -1 else (last_epoch + 1) * args.ipe
    for epoch in tqdm(range(args.epochs),
        desc="TRAINING: Epochs",
        dynamic_ncols=True,
        leave=False):

        batch_data_tr = Subset(data_tr, indices=kkm.pop_k(args.ex_per_epoch))
        batch_dataset = get_image_latent_dataset(model, batch_data_tr,
            latent_spec, args)
        loader = DataLoader(batch_dataset,
            batch_size=args.mini_bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=ImageLatentDataset.collate_fn)
        num_passes_over_loader = max(1, args.ipe // len(loader))
        batch_loader = chain(*[loader] * num_passes_over_loader)

        for x,z in tqdm(batch_loader,
            desc="TRAINING: Minibatches",
            dynamic_ncols=True,
            total=num_passes_over_loader * len(loader),
            leave=False):

            with torch.cuda.amp.autocast():
                loss = model(x, z, mask_ratio=args.mask_ratio)
                loss = torch.mean(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            scheduler.step()
            cur_step += 1

            if cur_step % 25 == 0:
                tqdm.write(f"{loss.item()}")

            if cur_step % log_iter == 0:
                wandb.log({"loss/vae_train": loss.item(),
                    "lr": scheduler.get_lr()},
                    step=cur_step)

        ########################################################################
        # Validate and save a checkpoint
        ########################################################################
        if epoch % args.eval_iter == 0:
            results = validate(model, data_tr, data_fn, data_te, args)
            if not os.path.exists(f"{save_dir}/images"):
                os.makedirs(f"{save_dir}/images")
            results["images/vae_train"].save(f"{save_dir}/images/{epoch+1}_train.png")
            results["images/vae_test"].save(f"{save_dir}/images/{epoch+1}_test.png")
            results["images/vae_train"] = wandb.Image(results["images/vae_train"])
            results["images/vae_test"] = wandb.Image(results["images/vae_test"])

            data_to_log = results | {"loss/vae_train": loss.item(),
                "lr": scheduler.get_lr()}

            tqdm.write(f"Epoch {epoch+1:4}/{args.epochs} | \
                lr {scheduler.get_lr():.5f} | \
                loss/vae_train {data_to_log['loss/vae_train']} | \
                loss/lin_probe_train {data_to_log['loss/lin_probe_train']} | \
                loss/lin_probe_test {data_to_log['loss/lin_probe_test']} | \
                accuracy/lin_probe_test {data_to_log['accuracy/lin_probe_test']:.3f}±{data_to_log['accuracy_lin_probe_test_conf']:.3f}")
        else:
            data_to_log = {"loss/pretrain": loss.item(),
                "lr": scheduler.get_lr()}
            tqdm.write(f"Epoch {epoch+1:4}/{args.epochs} | \
                lr {scheduler.get_lr():.5f} | \
                loss/vae_train {data_to_log['loss/vae_train']}")

        wandb.log(data_to_log, step=cur_step)
        torch.save({"model": model.cpu(), "optimizer": optimizer,
            "scheduler": scheduler, "args": args, "last_epoch": epoch,
            "kkm": kkm},
            f"{model_folder(args)}/{epoch+1}.pt")
        model = model.to(device)
