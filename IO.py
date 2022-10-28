import argparse
import os
from Utils import *

def argparse_file_type(x):
    return x if os.path.exists(x) or x.startswith("$") else False

def add_util_args(P):
    P.add_argument("--wandb", choices=["disabled", "online", "offline"], 
        default="online",
        help="Type of W&B logging")
    P.add_argument("--save_folder", default=f"{project_dir}/models",
        help="Folder inside which to save the enclosing folder for the results")
    P.add_argument("--suffix", type=str, default=None,
        help="Optional suffix")
    return P

def add_hardware_args(P):
    P.add_argument("--gpus", nargs="+", default=[0, 1], type=int,
        help="Device IDs of GPUs to use")
    P.add_argument("--num_workers", type=int, default=20,
        help="Number of DataLoader workers")
    return P

def add_linear_probe_args(P):
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
    P.add_argument("--probe_ignore_z",  type=int, default=1, choices=[0, 1],
        help="Whether to ignore the code in all linear probing")
    P.add_argument("--probe_eval_iter", type=int, default=-1,
        help="Evaluate linear probe every PROBE_EVAL_ITER epochs")
    return P

def add_train_imle_args(P):
    
    # Key arguments
    P.add_argument("--v_spec", nargs="*", default=[],
        help="Specification for making the autoencoder variational")
    P.add_argument("--arch", choices=["base_pretrained", "large_pretrained"], 
        default="base_pretrained",
        help="Type of ViT model to use")
    P.add_argument("--resume", default=None,
        help="Path to checkpoint to resume")
    P.add_argument("--finetune", choices=[0, 1], type=int, default=1,
        help="Whether to finetune an existing MAE model or train from scratch")
    P.add_argument("--ignore_z", choices=[0, 1], type=int, default=0,
        help="Whether to not use IMLE")
    P.add_argument("--data_tr", default="data/imagenet/train.tar", 
        type=argparse_file_type,
        help="String specifying training data")
    P.add_argument("--data_val", default="data/imagenet/val.tar", 
        type=argparse_file_type,
        help="String specifying finetuning data")
    
    # Evaluation arguments
    P.add_argument("--ex_for_eval_tr", default=8,
        help="Number of training examples for logging")
    P.add_argument("--ex_for_eval_te", default=8,
        help="Number of training examples for logging")
    P.add_argument("--z_per_ex_loss", default=128, type=int,
        help="Number of latents per example for logging losses")
    P.add_argument("--z_per_ex_vis", default=8, type=int,
        help="Number of latents per example for logging images")
    P.add_argument("--fast_linear_probe", default=1, choices=[0, 1], type=int,
        help="Whether to do fast linear probing each validation")
    
    # Logging arguments
    P.add_argument("--evals_per_epoch", type=int, default=1,
        help="Number of evaluations per epoch")
    P.add_argument("--save_iter", type=int, default=1,
        help="Number of epochs between each save")
    P.add_argument("--log_sampling", choices=[0, 1], type=int, default=1,
        help="Whether to log sampling data")

    # Shared training arguments between the MAE architecture and latent codes
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
    P.add_argument("--mask_ratio", type=float, default=.75,
        help="Mask ratio for the model")
    P.add_argument("--mini_bs", type=int, default=128,
        help="Batch size for training")
    P.add_argument("--norm_pix_loss", type=int, default=1, choices=[0, 1],
        help="Whether to predict normalized pixels")
    P.add_argument("--input_size", default=224, type=int,
        help="Size of (cropped) images to feed to the model")
    P.add_argument("--noise", default="gaussian", choices=["gaussian", "zeros"],
        help="Kind of noise to add inside the model")

    # Training arguments for the MAE architecture
    P.add_argument("--epochs", type=int, default=64,
        help="Number of sampling/training steps to run")
    P.add_argument("--lr", type=float, default=1e-4,
        help="Base learning rate")
    P.add_argument("--min_lr", type=float, default=0,
        help="Minumum learning rate")
    P.add_argument("--n_ramp", type=int, default=10,
        help="Number of epochs of linear learning rate warmup")

    # Training arguments for the latent code blocks
    P.add_argument("--epochs_z", type=int, default=16,
        help="Number of sampling/training steps to run")
    P.add_argument("--lr_z", type=float, default=1e-4,
        help="Base learning rate")
    P.add_argument("--min_lr_z", type=float, default=0,
        help="Minumum learning rate")
    P.add_argument("--n_ramp_z", type=int, default=10,
        help="Number of epochs of linear learning rate warmup")

    # Latent code block architectgure arguments
    P.add_argument("--act_type", default="leakyrelu",
        choices=["gelu", "leakyrelu"],
        help="Activation type")

    P.add_argument("--job_id", default=None,
        help="SLURM job_id if available")
    return P