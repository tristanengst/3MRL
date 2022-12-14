import argparse
import os
import random
from tqdm import tqdm
from Utils import *

import TrainIMLE

def unparse_args(args):
    """Returns [args] as a string that can be parsed again."""
    s = ""
    for k,v in vars(args).items():
        if isinstance(v, (list, tuple)):
            s += f" --{k} {' '.join([str(v_) for v_ in v])}"
        elif v is None:
            continue
        else:
            s += f" --{k} {v}"
    return s


def get_args_with_data_on_node(args, arg_names_to_move, out_dir="$SLURM_TMPDIR"):
    """Returns an (args, cmd) tuple where [args] is [args] modified to have the
    value in [args] of each element of [arg_names_to_move] listed inside
    [out_dir], and [cmd] is a string giving commands to move the files there.
    """
    s = ""
    args = vars(args)
    for a in arg_names_to_move:
        if a in args and isinstance(args[a], str) and os.path.exists(args[a]):
            s += f"rsync -rav --relative {args[a]} {out_dir}/\n"
            args[a] = f"{out_dir}/{args[a]}".replace("//", "/").strip("/")
        else:
            continue

    return argparse.Namespace(**args), f"{s}\n"

def get_time(hours):
    """Returns [hours] in SLURM string form.
    
    Args:
    hours   (int) -- the number of hours to run one chunk for
    """
    total_seconds = int(hours * 3600) - 1
    days = total_seconds // (3600 * 24)
    total_seconds = total_seconds % (3600 * 24)
    hours = total_seconds // 3600
    total_seconds = total_seconds % 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{days}-{hours}:{minutes}:{seconds}"

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("script",
        help="Script to run")
    P.add_argument("--time", type=float, default=3,
        help="Number of hours per task")
    P.add_argument("--parallel", type=int, default=1,
        help="Number of tasks to run in parallel")
    P.add_argument("--account", default="rrg-keli", choices=["def-keli", "rrg-keli"],
        help="ComputeCanada account to run on")
    P.add_argument("--end_chunk", default=0, type=int,
        help="Index of last chunk to run, with zero-indexing")
    P.add_argument("--start_chunk", default=0, type=int,
        help="Index of the chunk to resume from, ie. where there was an error")
    P.add_argument("--cluster", choices=["narval", "cedar", "beluga"], default="narval",
        help="Cluster on which the jobs are submitted")
    P.add_argument("--nproc_per_node", type=int, default=4,
        help="Number of GPUs per node")
    P.add_argument("--use_torch_distributed", type=int, choices=[0, 1], default=0,
        help="Whether to launch with 'python -m torch.distributed.launch' or with 'python'")
    slurm_args, unparsed_args = P.parse_known_args()

    ############################################################################
    # Figure out the commands to instantiate the required Python environment
    # based on the given cluster, the command to launch training on the compute
    # node(s), and initialize the command to move files onto the compute node.
    ############################################################################
    tqdm.write(f"Will use Python environment setup for cluster {slurm_args.cluster}")
    if slurm_args.cluster == "cedar":
        GPU_TYPE = "v100l"
    elif slurm_args.cluster == "narval":
        GPU_TYPE = "a100"
    elif slurm_args.cluster == "beluga":
        GPU_TYPE = "v100"
    else:
        raise ValueError(f"Unknown cluster {slurm_args.cluster}")

    if slurm_args.use_torch_distributed:
        launch_command = f"python -m torch.distributed.launch --nproc_per_node={slurm_args.nproc_per_node}"
    else:
        launch_command = "python"

    ############################################################################
    # Get script-specific argument settings
    ############################################################################
    if slurm_args.script == "LinearProbe.py":
        raise NotImplementedError()
    elif slurm_args.script == "TrainIMLE.py":
        args = TrainIMLE.get_args(unparsed_args)
        args, file_move_command = get_args_with_data_on_node(args, ["data_tr", "data_val"])

        NAME = f"model_{os.path.basename(TrainIMLE.model_folder(args, make_folder=False))}"
        NUM_GPUS = len(args.gpus)
        NUM_CPUS = min(24, max(1, NUM_GPUS) * 12)
        SCRIPT = f"{file_move_command}\n{launch_command} {slurm_args.script} {unparse_args(args)} --num_workers {NUM_CPUS} --save_folder '~/scratch/3MRL' --wandb offline"
        
        template = f"{os.path.dirname(__file__)}/slurm/slurm_template_sequential.txt"
        with open(template, "r") as f:
            template = f.read()
    else:
        raise ValueError(f"Unknown script '{slurm_args.script}")

    template = template.replace("SCRIPT", SCRIPT)
    template = template.replace("TIME", get_time(slurm_args.time))
    template = template.replace("NAME", NAME)
    template = template.replace("NUM_GPUS", str(NUM_GPUS))
    template = template.replace("NUM_CPUS", str(NUM_CPUS))
    template = template.replace("ACCOUNT", slurm_args.account)
    template = template.replace("GPU_TYPE", GPU_TYPE)
    slurm_script = f"slurm/{NAME}.sh"

    with open(slurm_script, "w+") as f:
        f.write(template)

    tqdm.write(f"File move command: {file_move_command}")
    tqdm.write(f"Launch command: {launch_command}")
    tqdm.write(f"Script:\n{SCRIPT}")
    tqdm.write(f"SLURM submission script written to {slurm_script}")
    tqdm.write(f"Outputs will write to job_results/{NAME}.txt")
    os.system(f"sbatch {slurm_script}")
