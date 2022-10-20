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
from FastLinearProbe import fast_linear_probe
from Models import *
from Utils import *

def model_folder(args, make_folder=False):
    """Returns the folder to which to save a model built with [args]."""
    data = os.path.basename(os.path.dirname(args.data_tr.strip("/"))).strip("/")
    v_spec = "_".join(args.v_spec)
    folder = f"{project_dir}/models/{data}-bs{args.ex_per_epoch}-epochs{args.epochs}-ipe{args.ipe}-lr{args.lr:.2e}-ns{tuple_to_str(args.ns)}-v_spec{v_spec}-{args.uid}{suffix_str(args)}"

    if make_folder:
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

def get_image_latent_dataset(model, dataset, latent_spec, args, epoch=0):
    """Returns an ImageLatent dataset constructed using the data in [dataset].
    This dataset can be used for one epoch's worth of training.

    Args:
    model       -- model to get codes for
    dataset     -- dataset containing the data to sample
    latent_spec -- dictionary of latent code shapes
    args        -- Namespace with --sp, --ns, and --code_bs arguments
    epoch       -- the index of the current epoch. Used only for logging
    """
    with torch.no_grad():
        least_losses = torch.ones(len(dataset), device=device) * float("inf")
        initial_codes = sample_latent_dict(latent_spec, len(dataset),
            device="cpu",
            noise=args.noise)
        best_latents = initial_codes["latents"]
        mask_noise = initial_codes["mask_noise"]
        latents_only_spec = {"latents": latent_spec["latents"]}

        loader = DataLoader(dataset,
            batch_size=args.code_bs,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=False)

        all_images = []
        for outer_idx,(x,_) in tqdm(enumerate(loader),
            desc="SAMPLING: Chunks of batch",
            total=len(loader),
            leave=False,
            dynamic_ncols=True):

            all_images.append(x)
            bs = len(x)
            start = outer_idx * args.code_bs
            stop = min(len(dataset), (outer_idx + 1) * args.code_bs)
            mask_noise_ = mask_noise[start:stop]

            for inner_idx in tqdm(range(args.ns // args.sp),
                desc=f"SAMPLING: code_bs {args.code_bs} | sp {args.sp}: Iterations over code batch size",
                leave=False,
                dynamic_ncols=True):

                latents = sample_latent_dict(latents_only_spec, bs * args.sp, noise=args.noise)
                z = {"mask_noise": mask_noise_} | latents
                with torch.cuda.amp.autocast():
                    losses = model(x, z,
                        mask_ratio=args.mask_ratio,
                        reduction="batch")
                _, idxs = torch.min(losses.view(bs, args.sp), dim=1)

                new_codes = z["latents"]
                new_codes = new_codes.view((bs, args.sp) + new_codes.shape[1:])
                new_codes = new_codes[torch.arange(bs), idxs]
                losses = losses.view(bs, args.sp)[torch.arange(bs), idxs]

                change_idxs = losses < least_losses[start:stop]
                best_latents[start:stop][change_idxs] = new_codes[change_idxs].cpu()

                if args.log_sampling and inner_idx > 0:
                    old_finite_idxs = ~torch.isinf(least_losses)
                    ll_old_mean = torch.mean(least_losses[old_finite_idxs])
                    
                    least_losses[start:stop][change_idxs] = losses[change_idxs]
                    ll_new_mean = torch.mean(least_losses[old_finite_idxs])
                                    
                    wandb.log({"pretrain/epoch": epoch,
                        "sampling/loss_mean": ll_new_mean,
                        "sampling/loss_delta": ll_new_mean - ll_old_mean, 
                        "sampling/step": epoch * len(loader) * args.ns // args.sp + outer_idx * args.ns // args.sp + inner_idx})
                else:
                    least_losses[start:stop][change_idxs] = losses[change_idxs]

    return ImageLatentDataset(torch.cat(all_images, axis=0).cpu(),
        mask_noises=mask_noise.cpu(),
        latents=best_latents)

def validate(model, data_tr, data_val, latent_spec, args):
    """Returns a dictionary of validation data about [model].

    Args:
    model       -- MaskedVAEViT model
    data_tr     -- Dataset of training data for reconstruction
    data_val    -- Dataset of data for linear probe training
    args        -- Namespace with relevant parameters
    """
    def get_reconstruction_images_loss(model, dataset, latent_spec, args, return_images=True, z_per_ex=8):
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

                bs_spec = {"mask_noise_bs": len(x),
                    "latents_bs": len(x) * z_per_ex}
                z = sample_latent_dict(latent_spec | bs_spec, noise=args.noise)
                with torch.cuda.amp.autocast():
                    loss, pred, mask = model(x, z,
                        mask_ratio=args.mask_ratio,
                        return_all=True,
                        ignore_z=args.ignore_z)

                total_loss += (loss.mean() * len(x)).detach()
                images.append(de_normalize(x).cpu())
                preds.append(pred.cpu())
                masks.append(mask.cpu())

        if return_images:
            images = torch.cat(images, dim=0)
            masks = torch.cat(masks, dim=0)
            preds = torch.cat(preds, dim=0)
            preds = preds.view(len(dataset), z_per_ex, *preds.shape[1:])
            image_grid = [[img] + [p for p in pred]
                for img,pred in zip(images, preds)]


        if return_images:
            return total_loss.item() / (len(dataset)), image_grid
        else:
            return total_loss.item() / (len(dataset))

    tqdm.write(f"---- VALIDATION | probe_ignore_z [{bool(args.probe_ignore_z)}] ----")
    if args.fast_linear_probe:
        classes = torch.linspace(0, len(data_tr.classes) - 1, args.val_n_way)
        classes = [data_tr.classes[int(c.item())] for c in classes]
        probe_acc = fast_linear_probe(model, data_tr, data_val, args,   
            classes=classes)
    else:
        probe_acc = -1

    vae_loss_te = get_reconstruction_images_loss(model,
        Subset(data_val, indices=random.sample(range(len(data_val)), k=512)),
        latent_spec, args,
        return_images=False,
        z_per_ex=args.z_per_ex_loss)

    idxs_tr = torch.linspace(0, len(data_tr) - 1, args.num_ex_for_eval_tr)
    _, images_tr = get_reconstruction_images_loss(model,
        Subset(data_tr, [int(idx) for idx in idxs_tr.tolist()]),
        latent_spec, args,
        return_images=True,
        z_per_ex=args.z_per_ex_vis)

    idxs_te = torch.linspace(0, len(data_val) - 1, args.num_ex_for_eval_te)
    _, images_te = get_reconstruction_images_loss(model,
        Subset(data_val, [int(idx) for idx in idxs_te.tolist()]),
        latent_spec, args,
        return_images=True,
        z_per_ex=args.z_per_ex_vis)

    return {
        "fast_linear_probe/acc_te": probe_acc,
        "pretrain/loss_te": vae_loss_te,
        "images/pretrain_train": images_to_pil_image(images_tr),
        "images/pretrain_test": images_to_pil_image(images_te)}

def get_args(args=None):

    def path_exists_type(x):
        return x if os.path.exists(x) or x.startswith("$") else False

    P = argparse.ArgumentParser()
    P.add_argument("--wandb", choices=["disabled", "online", "offline"], default="online",
        help="Type of W&B logging")
    P.add_argument("--suffix", type=str, default=None,
        help="Optional suffix")
    P.add_argument("--v_spec", nargs="*", default=[],
        help="Specification for making the autoencoder variational")
    P.add_argument("--arch", choices=["base_pretrained", "large_pretrained"], default="base_pretrained",
        help="Type of ViT model to use")
    P.add_argument("--resume", default=None,
        help="Path to checkpoint to resume")
    P.add_argument("--finetune", choices=[0, 1], type=int, default=1,
        help="Whether to finetune an existing MAE model or train from scratch")
    P.add_argument("--ignore_z", choices=[0, 1], type=int, default=0,
        help="Whether to use IMLE")
    # Data arguments
    P.add_argument("--data_tr", default="data/imagenet/train.tar", type=argparse_file_type,
        help="String specifying training data")
    P.add_argument("--data_val", default="data/imagenet/val.tar", type=argparse_file_type,
        help="String specifying finetuning data")

    # Evaluation arguments.
    P.add_argument("--num_ex_for_eval_tr", default=8,
        help="Number of training examples for logging")
    P.add_argument("--num_ex_for_eval_te", default=8,
        help="Number of training examples for logging")
    P.add_argument("--z_per_ex_loss", default=128, type=int,
        help="Number of latents per example for logging losses")
    P.add_argument("--z_per_ex_vis", default=8, type=int,
        help="Number of latents per example for logging images")
    P.add_argument("--fast_linear_probe", default=0, choices=[0, 1], type=int,
        help="Whether to do fast linear probing each validation")
    

    # Logging arguments
    P.add_argument("--eval_iter", type=int, default=1,
        help="Number of epochs between validation")
    P.add_argument("--save_iter", type=int, default=1,
        help="Number of epochs between validation")
    P.add_argument("--log_sampling", choices=[0, 1], type=int, default=1,
        help="Whether to log sampling data")

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
    P.add_argument("--ipe", type=int, default=10240,
        help="Gradient steps per epoch. Always at least the number of steps to see each minibatch once.")
    P.add_argument("--lr", type=float, default=1e-4,
        help="Base learning rate")
    P.add_argument("--min_lr", type=float, default=2e-5,
        help="Minumum learning rate")
    P.add_argument("--mask_ratio", type=float, default=.75,
        help="Mask ratio for the model")
    P.add_argument("--mini_bs", type=int, default=128,
        help="Batch size for training")
    P.add_argument("--n_ramp", type=int, default=10,
        help="Number of epochs of linear learning rate warmup")
    P.add_argument("--norm_pix_loss", type=int, default=1, choices=[0, 1],
        help="Whether to predict normalized pixels")
    P.add_argument("--input_size", default=224, type=int,
        help="Size of (cropped) images to feed to the model")
    P.add_argument("--noise", default="gaussian", choices=["gaussian", "zeros"],
        help="Kind of noise to add inside the model")

    # Interesting variational block arguments
    P.add_argument("--act_type", default="gelu", choices=["gelu"],
        help="Activation type")

    # Linear probe arguments
    P.add_argument("--global_pool", type=int, default=1, choices=[0, 1],
        help="Whether to global pool in linear probing")
    P.add_argument("--probe_bs", type=int, default=128,
        help="Linear probe training batch size")
    P.add_argument("--probe_bs_val", type=int, default=256,
        help="Linear probe test/data gathering batch size")
    P.add_argument("--val_n_way", type=int, default=10,
        help="Number of classes in probe/finetune data")
    P.add_argument("--val_n_shot", type=int, default=1000,
        help="Number of examples per class in probe/finetune data")
    P.add_argument("--probe_lr", type=float, default=1e-3,
        help="Linear probe base learning rate")
    P.add_argument("--probe_epochs", type=int, default=20,
        help="Linear probe number of epochs")
    P.add_argument("--probe_eval_iter", type=int, default=-1,
        help="Number of epochs between validation")
    P.add_argument("--probe_ignore_z",  type=int, default=1, choices=[0, 1],
        help="Whether to ignore the code in all linear probing")

    # Hardware arguments
    P.add_argument("--gpus", nargs="+", default=[0, 1], type=int,
        help="Device IDs of GPUs to use")
    P.add_argument("--job_id", default=None,
        help="SLURM job_id if available")
    P.add_argument("--num_workers", type=int, default=20,
        help="Number of DataLoader workers")

    args = P.parse_args() if args is None else P.parse_args(args)
    args.uid = wandb.util.generate_id() if args.job_id is None else args.job_id

    if len(args.v_spec) == 0:
        tqdm.write(f"WARNING: empty --v_spec precludes model from returning multiple outputs for one input. Consider adding a variational block with --noise set to 'zeros'")
        args.z_per_ex_vis = 1
        args.z_per_ex_loss = 1
    if args.sp > args.ns:
        tqdm.write(f"WARNING: --sp must be at most --ns. Setting --sp to --ns.")
        args.sp = args.ns
    return args

if __name__ == "__main__":
    args = get_args()
    pretty_print_args(args)

    ############################################################################
    # Load resumed things or instantiate them
    ############################################################################
    if args.resume is None:
        wandb.init(anonymous="allow", id=args.uid, config=args,
            mode=args.wandb, project="URSA",
            name=os.path.basename(model_folder(args)))

        if args.arch == "base_pretrained":
            mae_model_state = torch.load(f"{project_dir}/mae_checkpoints/mae_pretrain_vit_base_full.pth")["model"]
            mae = mae_vit_base_patch16(norm_pix_loss=args.norm_pix_loss)
        elif args.arch == "large_pretrained":
            mae_model_state = torch.load(f"{project_dir}/mae_checkpoints/mae_pretrain_vit_large_full.pth")["model"]
            mae = mae_vit_large_patch16(norm_pix_loss=args.norm_pix_loss)
        else:
            raise NotImplementedError()

        mae.load_state_dict(mae_model_state)
        model_kwargs = {"mae_model": mae} if args.finetune else mae.kwargs
        model = MaskedVAEViT(parse_variational_spec(args), **model_kwargs).to(device)

        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
        last_epoch = -1
    else:
        raise NotImplementedError()

    save_dir = model_folder(args)
    tqdm.write(f"LOG: Will save to {save_dir.replace(project_dir, '').strip('/')}")
    data_tr = data_path_to_dataset(args.data_tr,
        transform=get_train_transforms(args))
    data_val = data_path_to_dataset(args.data_val,
        transform=get_train_transforms(args))
    latent_spec = model.module.get_latent_spec(mask_ratio=args.mask_ratio,
        input_size=args.input_size)
    tqdm.write(f"LOG: Constructed latent shape dictionary: {latent_spec}")
    kkm = KOrKMinusOne(range(len(data_tr)), shuffle=True)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.epochs * args.ipe,
        warmup_steps=args.n_ramp * args.ipe,
        max_lr=args.lr,
        min_lr=args.min_lr,
        last_epoch=-1 if last_epoch == -1 else last_epoch * args.ipe)

    ############################################################################
    # Begin training
    ############################################################################
    if last_epoch == -1:
        results = validate(model, data_tr, data_val, latent_spec, args)
        conditional_safe_make_directory(f"{save_dir}/images")
        results["images/pretrain_train"].save(f"{save_dir}/images/0_train.png")
        results["images/pretrain_test"].save(f"{save_dir}/images/0_test.png")
        results["images/pretrain_train"] = wandb.Image(results["images/pretrain_train"])
        results["images/pretrain_test"] = wandb.Image(results["images/pretrain_test"])
        wandb.log(results | {"pretrain/epoch": 0, "pretrain/step": 0})
        tqdm.write(f"Epoch {0:4}/{args.epochs} | pretrain/loss_te {results['pretrain/loss_te']:.5f} | fast_linear_probe/acc_te {results['fast_linear_probe/acc_te']:.5f}")

    scaler = torch.cuda.amp.GradScaler()
    log_iter = max(1, int(args.epochs * args.ipe / 10000))
    tqdm.write(f"LOG: Will log every {log_iter} gradient steps")
    cur_step = 0 if last_epoch == -1 else (last_epoch + 1) * args.ipe
    for epoch in tqdm(range(args.epochs),
        desc="TRAINING: Epochs",
        dynamic_ncols=True,
        leave=False):

        batch_data_tr = Subset(data_tr, indices=kkm.pop_k(args.ex_per_epoch))
        batch_dataset = get_image_latent_dataset(model, batch_data_tr,
            latent_spec, args, epoch=epoch)
        loader = DataLoader(batch_dataset,
            batch_size=args.mini_bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=ImageLatentDataset.collate_fn,
            persistent_workers=True)

        num_passes_over_loader = max(1, args.ipe // len(loader))
        batch_loader = chain(*[loader] * num_passes_over_loader)
        gradient_steps = num_passes_over_loader * len(loader)
        tqdm.write(f"Epoch {epoch+1:4}/{args.epochs} | gradient steps {gradient_steps} | see an example {num_passes_over_loader} times")

        for x,z in tqdm(batch_loader,
            desc="TRAINING: Minibatches",
            dynamic_ncols=True,
            total=gradient_steps,
            leave=False):

            with torch.cuda.amp.autocast():
                loss = model(x, z,
                    mask_ratio=args.mask_ratio,
                    ignore_z=args.ignore_z)
                loss = torch.mean(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            scheduler.step()
            cur_step += 1

            if cur_step % log_iter == 0:
                wandb.log({"pretrain/loss_tr": loss.item(),
                    "pretrain/lr": scheduler.get_lr()[0],
                    "pretrain/step": cur_step})
                tqdm.write(f"\t{cur_step} loss_tr {loss.item()} | pretrain/lr {scheduler.get_lr()[0]:.5e}")


        ########################################################################
        # Validate and save a checkpoint
        ########################################################################
        if epoch % args.eval_iter == 0:
            results = validate(model, data_tr, data_val, latent_spec, args)
            conditional_safe_make_directory(f"{save_dir}/images")
            results["images/pretrain_train"].save(f"{save_dir}/images/{epoch+1}_train.png")
            results["images/pretrain_test"].save(f"{save_dir}/images/{epoch+1}_test.png")
            results["images/pretrain_train"] = wandb.Image(results["images/pretrain_train"])
            results["images/pretrain_test"] = wandb.Image(results["images/pretrain_test"])

            data_to_log = results | {"pretrain/loss_tr": loss.item(),
                "pretrain/lr": scheduler.get_lr()[0],
                "pretrain/epoch": epoch+1}

            tqdm.write(f"Epoch {epoch+1:4}/{args.epochs} | pretrain/lr {scheduler.get_lr()[0]:.5e} | pretrain/loss_tr {data_to_log['pretrain/loss_tr']:.5f} | pretrain/loss_te {data_to_log['pretrain/loss_te']:.5f} | fast_linear_probe/acc_te {data_to_log['fast_linear_probe/acc_te']:.5f}")
        else:
            data_to_log = {"loss/pretrain": loss.item(),
                "pretrain/lr": scheduler.get_lr()[0],
                "pretrain/epoch": epoch+1}
            tqdm.write(f"Epoch {epoch+1:4}/{args.epochs} | pretrain/lr {scheduler.get_lr()[0]:.5e} | pretrain/loss_tr {data_to_log['pretrain/loss_tr']:.5f} | fast_linear_probe/acc_te {data_to_log['fast_linear_probe/acc_te']:.5f}")
        
        wandb.log(data_to_log | {"pretrain/step": cur_step})
        
        if epoch % args.save_iter == 0 and args.save_iter > 0:
            conditional_safe_make_directory(f"{save_dir}")
            torch.save({"model": de_dataparallel(model).cpu().state_dict(),
                "encoder_kwargs": de_dataparallel(model).encoder_kwargs,
                "idx2v_method": de_dataparallel(model).idx2v_method,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler,
                "args": args,
                "last_epoch": epoch,
                "kkm": kkm},
                f"{model_folder(args)}/{epoch+1}.pt")
            model = model.to(device)
