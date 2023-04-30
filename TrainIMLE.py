import argparse
from copy import deepcopy
from itertools import chain
import time
from tqdm import tqdm
import wandb

import torch
from torch.optim import AdamW
import torch.nn
from torch.utils.data import DataLoader, Subset, Dataset

from original_code.models_mae import mae_vit_base_patch16 as foo

from Augmentation import *
from Data import *
from FastLinearProbe import fast_linear_probe
from IO import *
from Models import *
from Utils import *

def save_state(model, optimizer, scheduler, kkm, epoch, args, baseline=dict()):
    """Saves input training utilities."""
    torch.save({"model": de_dataparallel(model).cpu().state_dict(),
        "encoder_kwargs": de_dataparallel(model).encoder_kwargs,
        "idx2ip_method": de_dataparallel(model).idx2ip_method,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler,
        "args": args,
        "last_epoch": epoch,
        "kkm": kkm.state_dict(),
        "baseline": baseline},
        f"{model_folder(args, make_folder=True)}/{epoch}.pt")
    model = model.to(device)
    tqdm.write(f"Saved training state to {model_folder(args)}/{epoch}.pt")

def model_folder(args, make_folder=False):
    """Returns the folder to which to save a model built with [args].
    
    Args:
    args        -- argparse Namespace parameterizing the run being saved
    make_folder -- whether or not to create the folder corresponding to [args]
    """
    data = os.path.basename(os.path.dirname(args.data_tr.strip("/"))).strip("/")
    ip_spec = "_".join(args.ip_spec)
    job_id = "" if args.job_id is None else f"-{args.job_id}"
    lrs = "_".join([f"{lr:.2e}" for idx,lr in enumerate(args.lrs) if idx % 2 == 1])
    
    suffix_str = f"-{args.suffix}" if not args.suffix is None else ""
    folder = f"{args.save_folder}/models/{args.script}-{data}-bs{args.ex_per_epoch}-epochs{args.epochs}-ipe{args.ipe}-lr{lrs}-ns{args.ns}-ipspec{ip_spec}-{args.uid}{job_id}{suffix_str}"

    if make_folder:
        Utils.conditional_make_folder(folder)

    return folder

class ImageLatentDataset(Dataset):
    """Dataset for loading images and latents to a MaskedViTgen model. The
    provided collate function must be used in any DataLoaders wrappeding this
    dataset.

    Args:
    images          -- NxCxHxW tensor of images
    mask_codes      -- NxD tensor of mask noises
    latent_codes    -- Nx... tensor of latents for the model
    """
    def __init__(self, images, mask_codes, latent_codes):
        super(ImageLatentDataset, self).__init__()
        self.images = images
        self.mask_codes = mask_codes
        self.latent_codes = latent_codes

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.mask_codes[idx], self.latent_codes[idx]

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        mask_codes = torch.stack([b[1] for b in batch])
        latent_codes = torch.stack([b[2] for b in batch])
        return images, mask_codes, latent_codes

    @staticmethod    
    def get_image_latent_dataset(model, dataset, args, epoch=0):
        """Returns an ImageLatent dataset constructed using the data in [dataset].
        This dataset can be used for one epoch's worth of training.

        Args:
        model       -- model to get codes for
        dataset     -- dataset containing the data to sample
        args        -- Namespace with --sp, --ns, and --code_bs arguments
        epoch       -- the index of the current epoch. Used only for logging
        """
        start_time = time.time()
        with torch.no_grad():
            least_losses = torch.ones(len(dataset), device=device) * float("inf")
            best_codes = model.module.get_latent_codes(bs=len(dataset), device="cpu")
            mask_codes = model.module.get_mask_codes(bs=len(dataset), device="cpu")
            all_images = []
            
            # Tensor where the ith index gives the average loss of the first
            # --sp codes for that image. We use this in logging a useful proxy
            # for the diversity of results.
            first_losses = torch.ones(len(dataset)) * float("inf")

            loader = DataLoader(dataset,
                batch_size=args.code_bs,
                pin_memory=True,
                num_workers=args.num_workers,
                drop_last=False)

            for outer_idx,(x,_) in tqdm(enumerate(loader),
                desc="SAMPLING: Chunks of epoch",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                all_images.append(x)
                bs = len(x)
                start = outer_idx * args.code_bs
                stop = min(len(dataset), (outer_idx + 1) * args.code_bs)

                for inner_idx in tqdm(range(args.ns // args.sp),
                    desc=f"SAMPLING: code_bs {args.code_bs} | sp {args.sp} - Parallel Samples",
                    leave=False,
                    dynamic_ncols=True):

                    cur_latent_codes = model.module.get_latent_codes(bs=bs * args.sp)
                    
                    with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                        losses = model(x,
                            mask_codes=mask_codes[start:stop],
                            latent_codes=cur_latent_codes,
                            mask_ratio=args.mask_ratio,
                            reduction="batch").view(bs, args.sp)
                        least_losses_batch, idxs = torch.min(losses, dim=1)

                        new_codes = cur_latent_codes
                        new_codes = new_codes.view((bs, args.sp) + new_codes.shape[1:])
                        new_codes = new_codes[torch.arange(bs), idxs]

                        change_idxs = least_losses_batch < least_losses[start:stop]
                        best_codes[start:stop][change_idxs] = new_codes[change_idxs].cpu()
                        least_losses[start:stop][change_idxs] = least_losses_batch[change_idxs]
                        
                        # We log the mean and not the min because if args.sp is high
                        # enough, it will be hard to meaningfully improve on what we get
                        # the first time we sample, but we still want to see that the
                        # codes we use are better than average.
                        if inner_idx == 0:
                            first_losses[start:stop] = torch.mean(losses, dim=1).cpu()
        
        mean_first_loss = torch.mean(first_losses).item()
        mean_end_loss = torch.mean(least_losses).item()
        loss_delta = torch.mean(least_losses.cpu() - first_losses)
        wandb.log({"sampling/first_losses": mean_first_loss,
            "sampling/end_losses": mean_end_loss,
            "pretrain/step": epoch * args.ipe,
            "epoch": epoch})

        end_time = time.time()
        images_codes_per_sec = len(dataset) * args.ns / (end_time - start_time)
        sampling_loss_delta = torch.mean(least_losses.cpu() - first_losses)

        tqdm.write(f"SAMPLING: mean first loss {mean_first_loss:.5f} | mean end loss {mean_end_loss:.5f} | mean delta {sampling_loss_delta:.5f} | images codes per sec {images_codes_per_sec:.5f} ")

        return ImageLatentDataset(torch.cat(all_images, axis=0).cpu(),
            mask_codes=mask_codes.cpu(),
            latent_codes=best_codes)

def validate(model, data_tr, data_val, args, ignore_z=False):
    """Returns a dictionary of validation data about [model].

    Args:
    model       -- MaskedIPViT model
    data_tr     -- Dataset of training data for reconstruction
    data_val    -- Dataset of data for linear probe training
    args        -- Namespace with relevant parameters
    ignore_z    -- whether to ignore latent codes (treat as an autoencoder). If
                    True, overrides [args.ignore_z].
    """
    def get_reconstruction_images_loss(model, dataset, args, 
        return_images=True, codes_per_ex=8, val_ignore_z=False):
        """Returns an [loss] or a (loss, image_grid) tuple, where [loss] is the
        average loss of [model] on reconstructing images from [dataset] and
        [image_grid]  is a grid of images for qualitatively evaluating the 
        reconstructions.

        Args:
        model           -- MaskedIPViT model
        dataset         -- ImageFolder-like dataset with the images
        args            -- Namespace with relevant parameters
        return_images   -- return a (loss, image_grid) tuple or just loss
        codes_per_ex    -- number of codes to try for each image
        val_ignore_z    -- whether to ignore latent codes
        """
        # args.eval_bs * args.codes_per_ex is the maximum amount we can put on
        # GPUs. If [codes_per_ex] is actually smaller, we can increase the batch
        # size and run faster.
        loader = DataLoader(dataset,
            batch_size=args.eval_bs * max(1, args.z_per_ex_loss // codes_per_ex),
            num_workers=args.num_workers,
            pin_memory=True)

        images, targets, preds, inputs, total_loss = [], [], [], [], 0
        with torch.no_grad():
            for x,_ in tqdm(loader,
                desc="Validation: computing reconstruction loss and image grid",
                leave=False,
                dynamic_ncols=True):

                with torch.cuda.amp.autocast():
                    loss, pred, mask = model(x,
                        mask_ratio=args.mask_ratio,
                        codes_per_ex=codes_per_ex,
                        return_all=True,
                        ignore_z=val_ignore_z)

                total_loss += (loss.mean() * len(x)).detach()

                if return_images:
                    pred[(~mask.bool()).repeat_interleave(len(pred) // len(mask), dim=0)] = 1
                    if args.norm_pix_loss:
                        mean = x.mean(dim=-1, keepdim=True)
                        var = x.var(dim=-1, keepdim=True)
                        t = (x - mean) / (var + 1.e-6) ** .5
                    else:
                        t = x
                    
                    # What's here is de-normalized so it makes sense as an image.
                    # If the model predicts normalized images, many pixel values
                    # are negative! This also makes the task seem harder.
                    x = de_normalize(x)
                    inputs.append(torch.where(~mask.bool().cpu(), x, 1))
                    targets.append(torch.where(mask.bool().cpu(), de_normalize(t), 1))
                    images.append(x)
                    preds.append(pred.cpu())

        if return_images:
            images = torch.cat(images, dim=0)
            inputs = torch.cat(inputs, dim=0)
            targets = torch.cat(targets, dim=0)
            preds = torch.cat(preds, dim=0)
            preds = preds.view(len(dataset), codes_per_ex, *preds.shape[1:])
            image_grid = [[img] + [inp] + [t] + [p for p in pred]
                for img,inp,t,pred in zip(images, inputs, targets, preds)]
            
            return total_loss.item() / (len(dataset)), image_grid
        else:
            return total_loss.item() / (len(dataset))

    # Generally we will specify to not ignore latent codes in [args], but may
    # wish to ignore them once to get a baseline for what MAE would do.
    val_ignore_z = args.ignore_z | ignore_z

    if args.probe:
        probe_acc = fast_linear_probe(model, data_tr, data_val, args, verbose=False)
    else:
        probe_acc = -1

    idxs_te = Utils.sample(range(len(data_val)),
        k=min(args.ex_for_mse_loss, len(data_val)),
        seed=args.seed)
    gen_loss_te = get_reconstruction_images_loss(model,
        Subset(data_val, indices=idxs_te),
        args,
        return_images=False,
        codes_per_ex=args.z_per_ex_loss,
        val_ignore_z=val_ignore_z)

    idxs_tr = Utils.sample(range(len(data_tr)),
        k=min(args.ex_for_vis_tr, len(data_tr)),
        seed=args.seed)
    _, images_tr = get_reconstruction_images_loss(model,
        Subset(data_tr, indices=idxs_tr),
        args,
        return_images=True,
        codes_per_ex=args.z_per_ex_vis,
        val_ignore_z=val_ignore_z)
    
    idxs_te = Utils.sample(range(len(data_val)),
        k=min(args.ex_for_vis_te, len(data_val)),
        seed=args.seed)
    _, images_te = get_reconstruction_images_loss(model,
        Subset(data_val, indices=idxs_te),
        args,
        return_images=True,
        codes_per_ex=args.z_per_ex_vis,
        val_ignore_z=val_ignore_z)

    return {
        "fast_linear_probe/acc_te": probe_acc,
        "pretrain/loss_te": gen_loss_te,
        "images/pretrain_train": images_to_pil_image(images_tr),
        "images/pretrain_test": images_to_pil_image(images_te)}

def get_model_optimizer(args, resume_optimizer=False):
    """Returns a (model, optimizer) tuple where [model] is the model to be
    trained and [optimizer] is its optimizer given [args]. This function
    supports resuming old models and moving to the device.
    """
    model = get_masked_ipvit_model(args)
    if args.resume is not None:
        old_state = torch.load(args.resume)["model"]
        model.load_state_dict(old_state)

    model = nn.DataParallel(model, device_ids=args.gpus).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lrs[1], weight_decay=args.wd)
    
    if args.resume is not None and resume_optimizer:
        optimizer.load_state_dict(resume["optimizer"])
    
    return model, optimizer

def print_and_log_results(results, args, epoch=0, cur_step=0, baseline=False):
    """Prints and logs results [results].
    
    Args:
    results     -- dictionary of results to log. Keys containing the substring
                    'baseline' are logged and printed only if [baseline] is True
    args        -- Namespace with parameters for training
    epoch       -- the current epoch of training
    cur_step    -- the current gradient step of training
    baseline    -- whether the results being logged contain only baselines
    """
    save_dir = model_folder(args)
    Utils.conditional_make_folder(f"{save_dir}/images")
    cur_step = max(epoch * args.ipe, cur_step)

    # Save images and store them in a dictionary that will be logged later.
    images = {k: v for k,v in results.items() if "images" in k}
    log_images = {}
    for k,v in images.items():
        if "baseline" in k and baseline:
            # file_name = k.replace("images/", f"images/baseline_")
            # v.save(f"{save_dir}/{file_name}.png")
            log_images[k] = wandb.Image(v)
        elif "baseline" in k and not baseline:
            continue # This image has already been saved and logged
        else:
            file_name = k.replace("images/", f"images/{epoch+1}_{cur_step % args.ipe}")
            # v.save(f"{save_dir}/{file_name}.png")
            log_images[k] = wandb.Image(v)

    # Log the non-image results and the images that were saved earlier. Baseline
    # images are logged only if [baseline] is True.
    results = {k: v for k,v in results.items() if not "images" in k}
    wandb.log(results | {"pretrain/step": cur_step, "epoch": epoch} | log_images)

    if baseline:
        s = f"BASELINE {'-' * (19 + len(f'{args.epochs}{args.ipe * args.epochs}'))}"
        results = {k: v for k,v in results.items() if "baseline" in k}
        results = {k.replace("_baseline", ""): v for k,v in results.items()}
    else:
        s = f" EPOCH {epoch+1:4}/{args.epochs} -- Step {cur_step:6}/{args.ipe * args.epochs}"
        results = {k: v for k,v in results.items() if not "baseline" in k}

    for k,v in sorted(results.items()):
        s += f" | {k} {v:.3e}".replace("_params", "") if "lr" in k else ""
    for k,v in sorted(results.items(), reverse=True):
        s += f" | {k} {v:.5f}" if "loss" in k else ""
    for k,v in results.items():
        if "fast_linear_probe/acc_te" in k:
            s += f" | fast_linear_probe/acc_te {results[k]:.5f}"

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
    args.uid = wandb.util.generate_id() if args.uid is None else args.uid

    if ((args.probe_n_way > args.train_n_way)
        or args.probe_n_way == -1 and not args.train_n_way == -1):
        tqdm.write(f"Setting PROBE_N_WAY to {args.train_n_way} to match TRAIN_N_WAY")
        args.probe_n_way = args.train_n_way
    if ((args.probe_n_shot > args.train_n_shot)
        or args.probe_n_shot == -1 and not args.train_n_shot == -1):
        tqdm.write(f"Setting PROBE_N_SHOT to {args.train_n_shot} to match TRAIN_N_SHOT")
        args.probe_n_shot = args.train_n_shot

    if len(args.ip_spec) == 0:
        tqdm.write(f"WARNING: empty --ip_spec precludes model from returning multiple outputs for one input. Consider adding a variational block with --noise set to 'zeros'")
        
    if args.sp > args.ns:
        tqdm.write(f"WARNING: --sp must be at most --ns. Setting --sp to --ns.")
        args.sp = args.ns

    if args.ex_per_epoch // args.mini_bs > args.ipe:
        raise ValueError(f"Request at least --ex_per_epoch // --mini_bs iterations for --ipe")

    if args.script is None:
        args.script = "mae" if args.ignore_z else "mae-imle"
    else:
        if args.ignore_z and "imle" in args.script:
            raise ValueError(f"Should not run with --ignore_z set and 'imle' in the script.")

    args.lrs = Utils.StepScheduler.process_lrs(args.lrs)
    args.probe_lrs = Utils.StepScheduler.process_lrs(args.probe_lrs)

    return args

if __name__ == "__main__":
    args = get_args()
    Utils.set_seed(args.seed)

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
        baseline = {}
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

            model, optimizer = get_model_optimizer(args, resume_optimizer=True)
        else:
            args.arch = old_args.arch
            model, optimizer = get_model_optimizer(args, resume_optimizer=False)

        kkm = KOrKMinusOne.from_state_dict(resume["kkm"])
        last_epoch = resume["last_epoch"]
        last_step = last_epoch * old_args.ipe + old_args.ipe

        baseline = resume["baseline"] if "baseline" in resume else dict()
        
    tqdm.write(f"{Utils.sorted_namespace(args)}")
    tqdm.write(f"LOG: Will save to {model_folder(args)}")
    
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
   
    data_tr = get_fewshot_dataset(data_tr,
        n_way=args.train_n_way,
        n_shot=args.train_n_shot,
        seed=args.seed)
    data_val = get_fewshot_dataset(data_val,
        n_way=args.train_n_way,
        n_shot=args.train_n_shot,
        seed=args.seed,
        fewer_shots_if_needed=True)


    if kkm is None:
        kkm = KOrKMinusOne(range(len(data_tr)),
            shuffle=args.shuffle_data,
            seed=args.seed)

    tqdm.write(f"TRAINING DATA\n{data_tr}")
    tqdm.write(f"VALIDATION DATA\n{data_val}")
    tqdm.write(f"KKM\n{kkm}")
        
    scheduler = Utils.StepScheduler(optimizer, args.lrs,
        last_epoch=last_epoch,
        named_lr_muls={"mapping_net": 1 if args.mapping_net_eqlr else args.mapping_net_lrmul})
    tqdm.write(f"SCHEDULER\n{scheduler}")

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="3MRL", entity="apex-lab",
        name=os.path.basename(model_folder(args)),
        resume="allow" if args.continue_run else "never",
        settings=wandb.Settings(code_dir=os.path.dirname(__file__)))

    ############################################################################
    # If desired, evaluate with plain MAE. This is important as it gives the
    # baseline we want to improve from. We log these with a step of -1.
    ############################################################################
    if args.get_mae_baseline and args.resume is None:
        baseline = validate(model=nn.DataParallel(get_masked_ipvit_model(args),
            device_ids=args.gpus).to(device),
            data_tr=data_tr,
            data_val=data_val,
            args=args,
            ignore_z=True)
        baseline = {f"{k}_baseline": v for k,v in baseline.items()}

    if not baseline == {}:
        print_and_log_results(baseline, args,
            epoch=last_epoch + 1,
            baseline=True)

    ############################################################################
    # Begin training
    ############################################################################        
    scaler = torch.cuda.amp.GradScaler(init_scale=1, enabled=bool(args.fp16))
    log_iter = max(1, int((args.epochs - (last_epoch + 1)) * args.ipe / 1000))
    tqdm.write(f"LOG: Will log every {log_iter} gradient steps")
    
    for epoch in tqdm(range(last_epoch+1, args.epochs),
        desc="TRAINING: Epochs",
        dynamic_ncols=True,
        leave=False):

        # Do an initial validation of to see where what we need to improve from.
        # This should be equivalent to the last validation done in prior
        # training if we're resuming.
        if epoch == last_epoch + 1 and args.steps_per_eval > 0:
            data_to_log = validate(model=model,
                data_tr=data_tr,
                data_val=data_val,
                args=args)
            print_and_log_results(data_to_log | baseline, args,
                epoch=last_epoch,
                cur_step=last_epoch * args.ipe + args.ipe)

        # Sample latent codes and create the DataLoader for the epoch
        batch_dataset = ImageLatentDataset.get_image_latent_dataset(model,
            dataset=Subset(data_tr, indices=kkm.pop_k(args.ex_per_epoch)),
            args=args,
            epoch=epoch)
        loader = DataLoader(batch_dataset,
            batch_size=args.mini_bs,
            shuffle=args.shuffle_data,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=ImageLatentDataset.collate_fn,
            persistent_workers=True)
        
        num_passes_over_loader = max(1, args.ipe // len(loader))
        batch_loader = chain(*[loader] * num_passes_over_loader)
        gradient_steps = num_passes_over_loader * len(loader)

        # Print string signalling start of the epoch's training
        tqdm.write(f"Epoch {epoch+1:4}/{args.epochs} | gradient steps {gradient_steps} | each example appears {num_passes_over_loader} times")
        
        for batch_idx,(x,mask_codes,latent_codes) in tqdm(enumerate(batch_loader),
            desc="TRAINING: Minibatches",
            dynamic_ncols=True,
            total=gradient_steps,
            leave=False):

            with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                loss = model(x,
                    mask_codes=mask_codes,
                    latent_codes=latent_codes,
                    mask_ratio=args.mask_ratio,
                    ignore_z=args.ignore_z)
                loss = torch.mean(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()

            cur_step = epoch * args.ipe + batch_idx
            if cur_step % log_iter == 0 or batch_idx == gradient_steps - 1:
                data_to_log = {"pretrain/loss_tr": loss.item()}
                data_to_log |= {f"pretrain/lr": scheduler.get_lr()}
                print_and_log_results(data_to_log | baseline, args,
                    epoch=epoch,
                    cur_step=cur_step)

        data_to_log = {"pretrain/loss_tr": loss.item()}
        data_to_log |= {f"pretrain/lr": scheduler.get_lr()}
        data_to_log |= validate(model=model,
            data_tr=data_tr,
            data_val=data_val,
            args=args)
        print_and_log_results(data_to_log | baseline, args,
            epoch=epoch,
            cur_step=cur_step)

        if args.save_iter and args.epoch % args.save_iter or epoch == args.epochs - 1:
            save_state(model=model, optimizer=optimizer, scheduler=scheduler,
                kkm=kkm, epoch=epoch, args=args, baseline=baseline)
        elif args.save_iter == -1:
            # Delete previous saves
            save_state(model=model, optimizer=optimizer, scheduler=scheduler,
                kkm=kkm, epoch=epoch, args=args, baseline=baseline)

        scheduler.step()
            
