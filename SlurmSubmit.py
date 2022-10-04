"""File to use for submitting training scripts. Each run of this script will
submit a SLURM script corresponding to a singular run of a experiment script,
which handling SLURMification—in particular, chunking the job for ComputeCanada.

Because of the involved complexity and the fact that this project is extremely
compute-heavy, there's no support for hyperparameter tuning here; one must
submit a job for each desired hyperparameter configuration.

USAGE:
python SlurmSubmit.py Script.py --arg1 ...

It should be that deleting SlurmSubmit.py from the command yields exactly the
command desired for SLURM to run.
"""
import argparse
import os
import random
from Utils import *

def get_file_move_command(args):
    """Returns a (file_move_command, args) tuple, where [file_move_command] can
    be run to move files onto the compute node, and [args] is [args] but
    modified to use the local files.
    """
    key2file = {k: v for k,v in vars(args).items() if k.startswith("data_")}
    key2file_new = {k: f.replace(os.path.dirname(os.path.dirname(f)), "").lstrip("/")
        for k,f in key2file.items()}
    s = "\n".join([f"mkdir $SLURM_TMPDIR/{os.path.dirname(n)}\ncp {f} $SLURM_TMPDIR/{n}"
        for f,n in zip(key2file.values(), key2file_new.values())])

    key2file_new = {k: f"$SLURM_TMPDIR/{f}" for k,f in key2file_new.items()}
    args = argparse.Namespace(**(vars(args) | key2file_new))
    return s, args


def get_time(hours):
    """Returns [hours] in SLURM string form.
    Args:
    hours   (int) -- the number of hours to run one chunk for
    """
    total_seconds = (hours * 3600) - 1
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
        help="script to run")
    P.add_argument("--time", type=int, default=3,
        help="number of hours per task")
    P.add_argument("--parallel", type=int, default=1,
        help="number of hours per task")
    P.add_argument("--account", default="rrg-keli", choices=["def-keli", "rrg-keli"],
        help="number of hours per task")
    P.add_argument("--end_chunk", default=1, type=int,
        help="Index of last chunk to run, with zero-indexing")
    P.add_argument("--start_chunk", default=0, type=int,
        help="Index of the chunk to resume from, ie. where there was an error")
    P.add_argument("--cluster", choices=["narval", "cedar"], default="narval",
        help="Cluster on which the jobs are submitted")
    P.add_argument("--nproc_per_node", type=int, default=4,
        help="Number of GPUs")
    P.add_argument("--use_torch_distributed", type=int, choices=[0, 1], default=1,
        help="Number of GPUs")
    submission_args, unparsed_args = P.parse_known_args()


    ############################################################################
    # Figure out the commands to instantiate the required Python environment
    # based on the given cluster
    ############################################################################
    print(f"Will use Python environment setup for cluster {submission_args.cluster}")
    if submission_args.cluster == "cedar":
        # Recommended by CC, but I think a messy solution
        PYTHON_ENV_STR = "module load python/3.9\nvirtualenv $SLURM_TMPDIR/env\nsource $SLURM_TMPDIR/env/bin/activate\npip install --no-index --upgrade pip\npip install --no-index -r requirements.txt"

        # This is much cleaner
        PYTHON_ENV_STR = "module load python/3.9\nsource ~/py310URSA/bin/activate"
        GPU_TYPE = "v100l"
    elif submission_args.cluster == "narval":
        PYTHON_ENV_STR = "conda activate py310URSA"
        GPU_TYPE = "a100"
    else:
        raise ValueError(f"Unknown cluster {submission_args.cluster}")

    if submission_args.use_torch_distributed:
        launch_command = f"python -m torch.distributed.launch --nproc_per_node={submission_args.nproc_per_node}"
    else:
        launch_command = "python"

    ############################################################################
    # Get script-specific argument settings
    ############################################################################
    if submission_args.script.endswith("LinearProbe.py"):
        assert submission_args.parallel == 1
        from LinearProbe import get_args, get_linear_probe_folder
        args = get_args(unparsed_args)

        START_CHUNK = "0"
        END_CHUNK = "0" #str(args.epochs - 1)
        PARALLEL = "1"
        NUM_GPUS = str(submission_args.nproc_per_node)
        TIME = get_time(submission_args.time)
        NAME = get_linear_probe_folder(args).replace(f"{project_dir}/models/", "").replace("/", "_")

        file_move_command, args = get_file_move_command(args)

        unparsed_args_no_flag = [u.replace("--", "") for u in unparsed_args]
        
        unparsed_args = [f"--{k} {v}" for k,v in vars(args).items()
            if k in unparsed_args_no_flag]
        unparsed_args += [f"--world_size {submission_args.nproc_per_node}"]
        unparsed_args = " ".join(unparsed_args)

        SCRIPT = f"{file_move_command}\n{launch_command} {submission_args.script} {unparsed_args}"

        with open("slurm/slurm_template_sequential.txt", "r") as f:
            slurm_template = f.read()
    else:
        raise ValueError(f"Unknown script '{submission_args.script}")

    slurm_template = slurm_template.replace("SCRIPT", SCRIPT)
    slurm_template = slurm_template.replace("START_CHUNK", START_CHUNK)
    slurm_template = slurm_template.replace("END_CHUNK", END_CHUNK)
    slurm_template = slurm_template.replace("TIME", TIME)
    slurm_template = slurm_template.replace("NAME", NAME)
    slurm_template = slurm_template.replace("NUM_GPUS", NUM_GPUS)
    slurm_template = slurm_template.replace("PARALLEL", PARALLEL)
    slurm_template = slurm_template.replace("ACCOUNT", submission_args.account)
    slurm_template = slurm_template.replace("PYTHON_ENV_STR", PYTHON_ENV_STR)
    slurm_template = slurm_template.replace("GPU_TYPE", GPU_TYPE)
    slurm_script = f"slurm/{NAME}.sh"

    with open(slurm_script, "w+") as f:
        f.write(slurm_template)

    tqdm.write(f"Script:\n{SCRIPT}")
    tqdm.write(f"SLURM submission script written to {slurm_script}")
    os.system(f"sbatch {slurm_script}")
