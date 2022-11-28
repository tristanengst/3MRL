import argparse
import os
from Utils import *

def get_arg_names_from_fn(fn):
    P = argparse.ArgumentParser()
    P = fn(P)
    actions = [a for a in P._actions if isinstance(a, argparse._StoreAction)]
    return {a.dest for a in actions}


def argparse_file_type(x):
    if os.path.exists(x) or x.startswith("$"):
        return x
    else:
        raise FileNotFoundError(x)

def add_util_args(P):
    P.add_argument("--wandb", choices=["disabled", "online", "offline"], 
        default="online",
        help="Type of W&B logging")
    P.add_argument("--save_folder", default=".",
        help="Folder inside which to save the enclosing folder for the results")
    P.add_argument("--suffix", type=str, default=None,
        help="Optional suffix")
    P.add_argument("--notes", type=str, default=None,
        help="Optional notes for model significance")
    P.add_argument("--seed", type=int, default=1701,
        help="Random seed")
    P.add_argument("--uid", type=str, default=None,
        help="WandB UID")
    return P

def add_hardware_args(P):
    P.add_argument("--gpus", nargs="+", default=[0, 1], type=int,
        help="Device IDs of GPUs to use")
    P.add_argument("--num_workers", type=int, default=20,
        help="Number of DataLoader workers")
    P.add_argument("--fp16", default=True, action=argparse.BooleanOptionalAction,
        help="Whether to use FP16 or FP32 precision training. Sampling is always FP16 since we don't need gradients.")
    return P

def add_train_imle_debugging_args(P):
    P.add_argument("--shuffle_data", default=1, choices=[0, 1], type=int,
        help="Whether to shuffle data")
    P.add_argument("--use_augs", default=1, choices=[0, 1], type=int,
        help="Whether to use data augmentation")
    return P

def add_linear_probe_args(P):
    P.add_argument("--global_pool", type=int, default=1, choices=[0, 1],
        help="Whether to global pool in linear probing")
    P.add_argument("--probe_bs", type=int, default=16,
        help="Linear probe training batch size")
    P.add_argument("--probe_bs_val", type=int, default=256,
        help="Linear probe test/data gathering batch size")
    P.add_argument("--val_n_way", type=int, default=32,
        help="Number of classes in probe/finetune data")
    P.add_argument("--val_n_shot", type=int, default=100,
        help="Number of examples per class in probe/finetune data")
    P.add_argument("--probe_lr", type=float, default=1e-3,
        help="Linear probe base learning rate")
    P.add_argument("--probe_epochs", type=int, default=100,
        help="Linear probe number of epochs")
    P.add_argument("--probe_ignore_z",  type=int, default=1, choices=[0, 1],
        help="Whether to ignore the code in all linear probing")
    P.add_argument("--probe_eval_iter", type=int, default=-1,
        help="Evaluate every PROBE_EVAL_ITER epochs during training the probe")
    return P

def add_eval_imle_args(P):
    P.add_argument("--ex_for_mse_loss", type=int, default=2048,
        help="Number of examples for the fast linear probe")
    P.add_argument("--ex_for_vis_tr", default=8,
        help="Number of training examples for logging")
    P.add_argument("--ex_for_vis_te", default=8,
        help="Number of training examples for logging")
    P.add_argument("--z_per_ex_loss", default=128, type=int,
        help="Number of latents per example for logging losses")
    P.add_argument("--z_per_ex_vis", default=8, type=int,
        help="Number of latents per example for logging images")
    P.add_argument("--fast_linear_probe", default=1, choices=[0, 1], type=int,
        help="Whether to do fast linear probing each validation")
    
    # Logging arguments
    P.add_argument("--steps_per_eval", type=int, default=64,
        help="Number of evaluations per epoch")
    P.add_argument("--save_iter", type=int, default=-1,
        help="Number of epochs between each save")
    P.add_argument("--log_sampling", choices=[0, 1], type=int, default=1,
        help="Whether to log sampling data")
    P.add_argument("--get_mae_baseline", default=1, choices=[0, 1], type=int,
        help="Whether to evaluate without codes before training")
    return P

def add_train_imle_args(P):
    
    # Key arguments
    P.add_argument("--v_spec", nargs="*", default=[],
        help="Specification for making the autoencoder variational")
    P.add_argument("--arch", choices=["vit_base", "vit_large"], 
        default="vit_base",
        help="Type of ViT model to use")
    P.add_argument("--resume", default=None,
        help="Path to checkpoint to resume")
    P.add_argument("--finetune", choices=[0, 1], type=int, default=1,
        help="Whether to finetune an existing MAE model or train from scratch")
    P.add_argument("--ignore_z", choices=[0, 1], type=int, default=0,
        help="Whether to not use IMLE")
    P.add_argument("--fix_mask_noise", choices=[0, 1], type=int, default=0,
        help="Whether to fix the seed for generating masks to --seed")
    P.add_argument("--data_tr", default="../Data/imagenet/train", 
        type=argparse_file_type,
        help="String specifying training data")
    P.add_argument("--data_val", default="../Data/imagenet/val", 
        type=argparse_file_type,
        help="String specifying finetuning data")
    P.add_argument("--train_n_way", type=int, default=-1,
        help="Number of classes in generative modeling data")
    P.add_argument("--train_n_shot", type=int, default=-1,
        help="Number of examples per class in generative modeling data")

    # Shared training arguments between the MAE architecture and latent codes
    P.add_argument("--ex_per_epoch", type=int, default=2048,
        help="Number of examples to use in each sampling")
    P.add_argument("--code_bs", type=int, default=32,
        help="Batch size for sampling")
    P.add_argument("--ns", type=int, default=128,
        help="Number of latents from which to choose per image for sampling")
    P.add_argument("--sample_until_better_than_no_z", default=0, type=int,
        choices=[0, 1],
        help="Whether to sample until the loss is better than without a code")
    P.add_argument("--sp", type=int, default=64,
        help="Per-image latent code parallelism during sampling")
    P.add_argument("--ipe", type=int, default=64,
        help="Gradient steps per epoch. Always at least the number of steps to see each minibatch once. At least as worthwhile to tune as the learning rate.")
    P.add_argument("--mask_ratio", type=float, default=.75,
        help="Mask ratio for the model")
    P.add_argument("--mini_bs", type=int, default=32,
        help="Batch size for training")
    P.add_argument("--norm_pix_loss", type=int, default=1, choices=[0, 1],
        help="Whether to predict normalized pixels")
    P.add_argument("--input_size", default=224, type=int,
        help="Size of (cropped) images to feed to the model")
    P.add_argument("--noise", default="gaussian", choices=["gaussian", "zeros"],
        help="Kind of noise to add inside the model")
    P.add_argument("--beta1", type=float, default=.9,
        help="AdamW beta1")
    P.add_argument("--beta2", type=float, default=.95,
        help="AdamW beta2")
    P.add_argument("--wd", type=float, default=.01,
        help="AdamW weight decay")
    P.add_argument("--scheduler", default="constant",
        choices=["linear_ramp", "linear_ramp_cosine_decay", "constant"],
        help="Which scheduler to use")
    P.add_argument("--headstart_z", type=int, default=0,
        help="Number of epochs to only train z parameters")

    # Training arguments for the MAE architecture
    P.add_argument("--epochs", type=int, default=1024,
        help="Number of sampling/training steps to run")
    P.add_argument("--lr", type=float, default=1e-4,
        help="Base learning rate")
    P.add_argument("--min_lr", type=float, default=0,
        help="Minumum learning rate")
    P.add_argument("--n_ramp", type=int, default=0,
        help="Number of epochs of linear learning rate warmup")

    # Training arguments for the latent code blocks
    P.add_argument("--lr_z", type=float, default=1e-4,
        help="Base learning rate")

    # Latent code block architectgure arguments
    P.add_argument("--act_type", default="leakyrelu",
        choices=["gelu", "leakyrelu"],
        help="Activation type")

    P.add_argument("--job_id", default=None,
        help="SLURM job_id if available, or UID of run to resume.")
    return P