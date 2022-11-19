import argparse
from itertools import chain
from tqdm import tqdm
import wandb

# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torch
from torch.optim import AdamW
import torch.nn
from torch.utils.data import DataLoader, Subset, Dataset

from original_code.models_mae import mae_vit_base_patch16 as foo

from Augmentation import *
import Misc
from Data import *
from FastLinearProbe import fast_linear_probe
from IO import *
from Models import *
from Utils import *

def model_folder(args, make_folder=False):
    """Returns the folder to which to save a model built with [args].
    
    Args:
    args        -- argparse Namespace parameterizing the run being saved
    make_folder -- whether or not to create the folder corresponding to [args]
    """
    data = os.path.basename(os.path.dirname(args.data_tr.strip("/"))).strip("/")
    v_spec = "_".join(args.v_spec)
    folder = f"{args.save_folder}/{data}-{args.arch}-bs{args.ex_per_epoch}-epochs{args.epochs}-headstartz{args.headstart_z}-ipe{args.ipe}-lr{args.lr:.2e}-lrz{args.lr_z}-nramp{args.n_ramp}-ns{args.ns}-scheduler{args.scheduler}-vspec{v_spec}-{args.uid}{Misc.suffix_str(args)}"

    if make_folder:
        Misc.conditional_safe_make_directory(folder)
        if not os.path.exists(f"{folder}/config.json"):
            with open(f"{folder}/config.json", "w+") as f:
                json.dump(vars(args), f)
    return folder

class ImageLatentDataset(Dataset):
    """Dataset for loading images and latents to a MaskedViTgen model. The
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

    def get_val_results(self, model, args):
        """Returns a dictionary giving the loss and image grid given from 
        [model] for a small subset of the images and latents in [self]. This is
        useful as a quick validation.
        
        Args:
        model   -- model to use
        args    -- argparse Namespace
        """
        xz = [self.__getitem__(idx) for idx in range(min(args.mini_bs, len(self)))]
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x,z = ImageLatentDataset.collate_fn(xz)
                loss = model(x, z,
                    mask_ratio=args.mask_ratio,
                    ignore_z=args.ignore_z)
        
        xz = [self.__getitem__(idx) for idx in range(min(args.ex_for_vis_tr, len(self)))]
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x,z = ImageLatentDataset.collate_fn(xz)
                _, pred, _ = model(x, z,
                    mask_ratio=args.mask_ratio,
                    return_all=True,
                    ignore_z=args.ignore_z)
        
        loss = loss.mean().item()
        images = de_normalize(x).cpu()
        preds = pred.cpu()
        image_grid = [[image, pred] for image,pred in zip(images, preds)]
        return {"image_grid": image_grid, "loss": loss}
            
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
        initial_codes = sample_latent_dict(latent_spec,
            bs=len(dataset),
            device="cpu",
            noise=args.noise)
        best_latents = initial_codes["latents"]
        mask_noise = initial_codes["mask_noise"]
        latents_only_spec = {"latents": latent_spec["latents"]}

        # For logging
        first_losses = []

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

            for inner_idx in tqdm(range(args.ns // args.sp),
                desc=f"SAMPLING: code_bs {args.code_bs} | sp {args.sp}: Iterations over code batch size",
                leave=False,
                dynamic_ncols=True):

                latents = sample_latent_dict(latents_only_spec,
                    bs=bs * args.sp,
                    noise=args.noise)
                z = {"mask_noise": mask_noise[start:stop]} | latents
                
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
                least_losses[start:stop][change_idxs] = losses[change_idxs]
                
                # We log the mean and not the min because if args.sp is high
                # enough, it will be hard to meaningfully improve on what we get
                # the first time we sample, but we still want to see that the
                # codes we use are better than average.
                if inner_idx == 0 and args.log_sampling:
                    first_losses.append(losses.mean().item())

    if args.log_sampling:
        wandb.log({"sampling/first_losses": np.mean(first_losses),
            "sampling/end_losses": least_losses.mean().item(),
            "pretrain/step": epoch * args.ipe,
            "epoch": epoch})

    tqdm.write(f"SAMPLING: average first loss {np.mean(first_losses):.5f} | average end loss {least_losses.mean().item():.5f} ")

    return ImageLatentDataset(torch.cat(all_images, axis=0).cpu(),
        mask_noises=mask_noise.cpu(),
        latents=best_latents)

def validate(model, data_tr, data_val, latent_spec, args, ignore_z=False):
    """Returns a dictionary of validation data about [model].

    Args:
    model       -- MaskedIPViT model
    data_tr     -- Dataset of training data for reconstruction
    data_val    -- Dataset of data for linear probe training
    args        -- Namespace with relevant parameters
    ignore_z    -- whether to ignore latent codes (treat as an autoencoder). If
                    True, overrides [args.ignore_z].
    """
    def get_reconstruction_images_loss(model, dataset, latent_spec, args, 
        return_images=True, z_per_ex=8, val_ignore_z=False):
        """Returns an [loss] or a (loss, image_grid) tuple, where [loss] is the
        average loss of [model] on reconstructing images from [dataset] and
        [image_grid]  is a grid of images for qualitatively evaluating the 
        reconstructions.

        Args:
        model           -- MaskedIPViT model
        dataset         -- ImageFolder-like dataset with the images
        latent_spec     -- latent specification for [model]
        args            -- Namespace with relevant parameters
        return_images   -- return a (loss, image_grid) tuple or just loss
        z_per_ex        -- number of codes to try for each image
        val_ignore_z    -- whether to ignore latent codes
        """
        loader = DataLoader(dataset,
            batch_size=args.code_bs * max(1, args.sp // z_per_ex),
            num_workers=args.num_workers,
            pin_memory=True)

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
                        ignore_z=val_ignore_z)

                total_loss += (loss.mean() * len(x)).detach()

                if return_images:
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
            
            return total_loss.item() / (len(dataset)), image_grid
        else:
            return total_loss.item() / (len(dataset))

    # Generally we will specify to not ignore latent codes in [args], but may
    # wish to ignore them once to get a baseline for what MAE would do.
    val_ignore_z = args.ignore_z | ignore_z

    if args.fast_linear_probe:
        probe_acc = fast_linear_probe(model, data_tr, data_val, args, verbose=False)
    else:
        probe_acc = -1

    idxs_te = Misc.sample(range(len(data_val)),
        k=min(args.ex_for_mse_loss, len(data_val)),
        seed=args.seed)
    gen_loss_te = get_reconstruction_images_loss(model,
        Subset(data_val, indices=idxs_te),
        latent_spec, args,
        return_images=False,
        z_per_ex=args.z_per_ex_loss,
        val_ignore_z=val_ignore_z)

    idxs_tr = Misc.sample(range(len(data_tr)),
        k=args.ex_for_vis_tr,
        seed=args.seed)
    _, images_tr = get_reconstruction_images_loss(model,
        Subset(data_tr, indices=idxs_tr),
        latent_spec, args,
        return_images=True,
        z_per_ex=args.z_per_ex_vis,
        val_ignore_z=val_ignore_z)
    
    idxs_te = Misc.sample(range(len(data_val)),
        k=args.ex_for_vis_te,
        seed=args.seed)
    _, images_te = get_reconstruction_images_loss(model,
        Subset(data_val, indices=idxs_te),
        latent_spec, args,
        return_images=True,
        z_per_ex=args.z_per_ex_vis,
        val_ignore_z=val_ignore_z)

    return {
        "fast_linear_probe/acc_te": probe_acc,
        "pretrain/loss_te": gen_loss_te,
        "images/pretrain_train": images_to_pil_image(images_tr),
        "images/pretrain_test": images_to_pil_image(images_te)}


def save_state(model, optimizer, scheduler, kkm, epoch, args):
    """Saves input training utilities."""
    torch.save({"model": de_dataparallel(model).cpu().state_dict(),
        "encoder_kwargs": de_dataparallel(model).encoder_kwargs,
        "idx2v_method": de_dataparallel(model).idx2v_method,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler,
        "args": args,
        "last_epoch": epoch,
        "kkm": kkm.state_dict()},
        f"{model_folder(args, make_folder=True)}/{epoch}.pt")
    model = model.to(device)
    tqdm.write(f"Saved training state to {model_folder(args)}/{epoch}.pt")

def print_and_log_results(data_to_log, args, epoch=0, cur_step=0):
    """Prints and logs results [data_to_log].

    Args:
    data_to_log -- dictionary of things to log
    args        -- argparse Namespace for training
    epoch       -- index of the current epoch or 'baseline'
    cur_step    -- the current global training gradient step
    """
    if epoch == "baseline":
        s = f"BASELINE {'-' * (19 + len(f'{args.epochs}{args.ipe * args.epochs}'))}"
        image_id_str = "baseline"
    else:
        s = f" EPOCH {epoch+1:4}/{args.epochs} -- Step {cur_step:6}/{args.ipe * args.epochs}"
        image_id_str = f"{epoch+1}_{cur_step % args.ipe}"
    
    save_dir = model_folder(args)
    Misc.conditional_safe_make_directory(f"{save_dir}/images")

    # Save images and convert them to wandb.Image format
    for k in data_to_log:
        if "images" in k:
            file_name = k.replace("images/", f"images/{image_id_str}")
            data_to_log[k].save(f"{save_dir}/{file_name}.png")
            data_to_log[k] = wandb.Image(data_to_log[k])

    # Upload to WandB
    if epoch == "baseline":
        wandb.log(data_to_log | {"pretrain/step": -1})
    else:
        wandb.log(data_to_log | {"pretrain/step": cur_step, "epoch": epoch})

    # Print string describing training
    for k,v in sorted(data_to_log.items()):
        s += f" | {k} {v:.3e}".replace("_params", "") if "lr" in k else ""
    for k,v in sorted(data_to_log.items(), reverse=True):
        s += f" | {k} {v:.5f}" if "loss" in k else ""
    
    if "fast_linear_probe/acc_te" in data_to_log:
        s += f" | fast_linear_probe/acc_te {data_to_log['fast_linear_probe/acc_te']:.5f}"

    s = s.replace("pretrain/", "")
    tqdm.write(s)
    

def get_args(args=None):
    """Returns parsed arguments, either from [args] or standard input."""
    P = argparse.ArgumentParser()
    P = add_hardware_args(P)
    P = add_linear_probe_args(P)
    P = add_train_imle_args(P)
    P = add_util_args(P)
    P = add_eval_imle_args(P)
    P = add_train_imle_debugging_args(P)

    args = P.parse_args() if args is None else P.parse_args(args)
    args.save_folder = args.save_folder.strip("/")

    if args.uid is None:
        args.uid = wandb.util.generate_id()

    if args.scheduler == "constant" and not args.n_ramp == 0:
        raise ValueError(f"Can not ramp constant scheduler. Set --n_ramp to zero")
    if args.scheduler == "linear_ramp_cosine_decay" and not args.headstart_z == 0:
        raise ValueError(f"Can not headstart the mapping network with this scheduler")

    if len(args.v_spec) == 0:
        tqdm.write(f"WARNING: empty --v_spec precludes model from returning multiple outputs for one input. Consider adding a variational block with --noise set to 'zeros'")
        args.z_per_ex_vis = 1
        args.z_per_ex_loss = 1
    if args.sp > args.ns:
        tqdm.write(f"WARNING: --sp must be at most --ns. Setting --sp to --ns.")
        args.sp = args.ns

    if args.ex_per_epoch // args.mini_bs > args.ipe:
        raise ValueError(f"Request at least --ex_per_epoch // --mini_bs iterations for --ipe")

    return args

def get_masked_ipvit_model(args):
    """Returns an MaskedIPViT model with the weights and architecture of an MAE
    model specified in [args. Resuming a MaskedIPViT model is not handeled, ie.
    the model weights are pure MAE weights.
    """
    # Instantiate the MAE model we're finetuning
    if args.arch == "vit_base":
        mae_model_state = torch.load(f"{os.path.dirname(__file__)}/mae_checkpoints/mae_pretrain_vit_base_full.pth")["model"]
        mae = mae_vit_base_patch16(norm_pix_loss=args.norm_pix_loss)
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

def get_model_optimizer(args):
    """Returns a (model, optimizer) tuple where [model] is the model to be
    trained and [optimizer] is its optimizer given [args]. This function
    supports resuming old models and moving to the device.
    """
    model = get_masked_ipvit_model(args)
    if args.resume is not None:
        old_state = torch.load(args.resume)["model"]
        model.load_state_dict(old_state)

    model = nn.DataParallel(model, device_ids=args.gpus).to(device)
    
    params_z = [v for p,v in model.named_parameters()
            if "v_method" in p]
    params_mae = [v for p,v in model.named_parameters()
            if not "v_method" in p]
    optimizer = AdamW([{"params": params_z,
        "lr": args.lr_z,
        "initial_lr": 0,
        "name": "params_z"},
        {"params": params_mae,
        "lr": args.lr,
        "initial_lr": 0,
        "name": "params_mae"}],
        betas=(args.beta1, args.beta2),
        weight_decay=args.wd)
    
    if args.resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
    
    return model, optimizer

if __name__ == "__main__":
    args = get_args()
    Misc.set_seed(args.seed)

    ############################################################################
    # Load resumed things or instantiate them. If resuming and the resumed
    # model's UID is given, all but evaluation- and hardware-specific arguments
    # are loaded along with the old model's KKM. Otherwise when resuming, only
    # the old model and optimizer states, and last-run epoch index are loaded.
    ############################################################################
    if args.resume is None:
        model, optimizer = get_model_optimizer(args)
        wandb.init(anonymous="allow", id=args.uid, config=args,
            mode=args.wandb, project="3MRL",
            name=os.path.basename(model_folder(args)))
        kkm = None
        last_epoch = -1
        last_step = -1
    else:
        resume = torch.load(args.resume)
        old_args = resume["args"]

        if old_args.uid == args.uid:
            # Update [args] to be identical to [old_args] except in respect to
            # hardware and evaluation.
            keep_args = get_arg_names_from_fn(add_hardware_args)
            keep_args |= get_arg_names_from_fn(add_linear_probe_args)
            keep_args |= get_arg_names_from_fn(add_eval_imle_args)
            keep_args |= {"save_folder"}
            keep_args = {k: v for k,v in vars(args).items() if k in keep_args}
            args.__dict__.update(vars(old_args) | keep_args)

            model, optimizer = get_model_optimizer(args)
            kkm = KOrKMinusOne.from_state_dict(resume["kkm"])
            wandb.init(id=args.uid, resume="must", mode=args.wandb,
                project="3MRL", config=args,
                name=os.path.basename(model_folder(args)))
        else:
            args.arch = old_args.arch
            model, optimizer = get_model_optimizer(args)
            wandb.init(anonymous="allow", id=args.uid, config=args,
                mode=args.wandb, project="3MRL",
                name=os.path.basename(model_folder(args)))

        last_epoch = resume["last_epoch"]
        last_step = last_epoch * old_args.ipe + old_args.ipe

    latent_spec = model.module.get_latent_spec(
        mask_ratio=args.mask_ratio,
        input_size=args.input_size)
        
    Misc.pretty_print_args(args)
    tqdm.write(f"LOG: Will save to {model_folder(args)}")
    
    # tqdm.write(f"MODEL\n{dict(model.named_parameters()).keys()}")
    tqdm.write(f"OPTIMIZER\n{optimizer}")
    tqdm.write(f"LATENT_SPEC\n{latent_spec}")

    ############################################################################
    # Get the data and KKM. If resuming from an MAE checkpoint, we need to train
    # on all of ImageNet or any improvements can be attributed to overfitting
    # the subsample we train on. However, for debugging or training from scratch
    # a subsample of ImageNet can be used.
    ############################################################################
    data_tr = data_path_to_dataset(args.data_tr,
        transform=get_train_transforms(args))
    data_val = data_path_to_dataset(args.data_val,
        transform=get_train_transforms(args))
   
    if not args.train_n_way == -1 or not args.train_n_shot == -1:
        if args.train_n_way == -1:
            classes = -1
        else:
            classes = torch.linspace(0, len(data_tr.classes) - 1, args.train_n_way)
            classes = [data_tr.classes[int(c.item())] for c in classes]

        data_tr = get_fewshot_dataset(data_tr,
            n_way=args.train_n_way,
            n_shot=args.train_n_shot,
            classes=classes,
            seed=args.seed)
        data_val = get_fewshot_dataset(data_val,
            n_way=args.train_n_way,
            n_shot=args.train_n_shot,
            classes=classes,
            seed=args.seed,
            fewer_shots_if_needed=True)

    if kkm is None:
        kkm = KOrKMinusOne(range(len(data_tr)),
            shuffle=args.shuffle_data,
            seed=args.seed)

    tqdm.write(f"TRAINING DATA\n{data_tr}")
    tqdm.write(f"VALIDATION DATA\n{data_val}")
    tqdm.write(f"KKM\n{kkm}")
        
    ############################################################################
    # Get the scheduler
    ############################################################################    
    if args.scheduler == "linear_ramp_cosine_decay":
        scheduler = CosineAnnealingWarmupRestartsMultipleParamGroups(optimizer,
            first_cycle_steps=args.epochs * args.ipe,
            warmup_steps=args.n_ramp * args.ipe,
            min_lr=args.min_lr,
            last_epoch=last_step)
    elif args.scheduler == "linear_ramp":
        scheduler = LinearRampScheduler(optimizer,
            warmup_steps=args.n_ramp * args.ipe,
            min_lr=args.min_lr,
            pg2base_lrs={"params_mae": args.lr, "params_z": args.lr_z},
            pg2start_step={"params_mae": args.headstart_z * args.ipe, "params_z": 0},
            last_epoch=last_step)
    elif args.scheduler == "constant":
        scheduler = NoChangeScheduler(optimizer, last_epoch=last_epoch)
    else:
        raise NotImplementedError()

    tqdm.write(f"SCHEDULER\n{scheduler}")

    ############################################################################
    # If desired, evaluate with plain MAE. This is important as it gives the
    # baseline we want to improve from. We log these with a step of -1.
    ############################################################################
    if args.get_mae_baseline:
        mae = nn.DataParallel(get_masked_ipvit_model(args), device_ids=args.gpus)
        mae_results = validate(model=mae.to(device),
            data_tr=data_tr,
            data_val=data_val,
            latent_spec=latent_spec,
            args=args,
            ignore_z=True)
        print_and_log_results(mae_results, args, epoch="baseline")

    ############################################################################
    # Begin training
    ############################################################################        
    scaler = torch.cuda.amp.GradScaler(init_scale=1)
    log_iter = max(1, int((args.epochs - (last_epoch + 1)) * args.ipe / 10000))
    tqdm.write(f"LOG: Will log every {log_iter} gradient steps")
    
    for epoch in tqdm(range(last_epoch+1, args.epochs),
        desc="TRAINING: Epochs",
        dynamic_ncols=True,
        leave=False):

        # Do an initial validation of to see where what we need to improve from.
        # This should be equivalent to the last validation done in prior
        # training if we're resuming.
        if epoch == last_epoch + 1:
            data_to_log = validate(model=model,
                data_tr=data_tr,
                data_val=data_val,
                latent_spec=latent_spec,
                args=args)
            print_and_log_results(data_to_log, args, epoch=last_epoch)

        # Sample latent codes and create the DataLoader for the epoch
        batch_data_tr = Subset(data_tr, indices=kkm.pop_k(args.ex_per_epoch))
        batch_dataset = get_image_latent_dataset(model, batch_data_tr,
            latent_spec, args, epoch=epoch)
        loader = DataLoader(batch_dataset,
            batch_size=args.mini_bs,
            shuffle=args.shuffle_data,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=ImageLatentDataset.collate_fn,
            persistent_workers=True)
        num_passes_over_loader = max(1, args.ipe // len(loader))
        batch_loader = chain(*[loader] * num_passes_over_loader)

        # Print string signalling start of the epoch's training
        gradient_steps = num_passes_over_loader * len(loader)
        tqdm.write(f"Epoch {epoch+1:4}/{args.epochs} | gradient steps {gradient_steps} | each example appears {num_passes_over_loader} times")

        ########################################################################
        # Train for one epoch on the data
        ########################################################################
        pre_epoch_results = batch_dataset.get_val_results(model, args=args)
        
        for batch_idx,(x,z) in tqdm(enumerate(batch_loader),
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

            cur_step = epoch * args.ipe + batch_idx

            if (cur_step % log_iter == 0
                or batch_idx == gradient_steps - 1):
                
                data_to_log = {"pretrain/loss_tr": loss.item()}
                data_to_log |= {f"pretrain/lr_{g}": lr
                    for g,lr in scheduler_to_lrs(scheduler).items()}
                print_and_log_results(data_to_log, args,
                    epoch=epoch,
                    cur_step=cur_step)
            elif (cur_step % args.steps_per_eval == 0
                and not (batch_idx == 0 and epoch == last_epoch + 1)):

                data_to_log = {"pretrain/loss_tr": loss.item()}
                data_to_log |= {f"pretrain/lr_{g}": lr
                    for g,lr in scheduler_to_lrs(scheduler).items()}
                data_to_log |= validate(model=model,
                        data_tr=data_tr,
                        data_val=data_val,
                        latent_spec=latent_spec,
                        args=args)
                print_and_log_results(data_to_log, args,
                    epoch=epoch,
                    cur_step=cur_step)

        post_epoch_results = batch_dataset.get_val_results(model, args=args)
        start_image_gid = pre_epoch_results["image_grid"]
        end_image_gid = post_epoch_results["image_grid"]

        image_grid = [[img, s, e]
            for (img,s),(_,e) in zip(start_image_gid, end_image_gid)]
        wandb.log({"epoch": epoch,
            "epoch/loss_end": post_epoch_results["loss"],
            "epoch/image_grid": wandb.Image(images_to_pil_image(image_grid)),
            "epoch/loss_start": pre_epoch_results["loss"]})
        
        if ((epoch % args.save_iter == 0
            or epoch == args.epochs - 1)
            and args.save_iter > -1):
            save_state(model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                kkm=kkm,
                epoch=epoch,
                args=args)
            
