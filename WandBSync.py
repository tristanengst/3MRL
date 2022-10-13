"""The latest weapon for WandB on Narval: automated offline job syncing!

USAGE: python WandBSync.py

Notes:
- If you want to remove a run—possibly split into one or more chunks—you must
    do the following in the order presented to avoid breaking WandB:
    1. Stop the run on on ComputeCanada
    2. Run this script
    3. Remove the run on WandB
"""
import os
from tqdm import tqdm
import shutil

files = [f"wandb/{f}" for f in os.listdir("wandb") if f.startswith("offline-run")]
for f in tqdm(files):
    # The documentation of subprocess is grossly overcomplicated
    os.system(f"wandb sync {f} > wandb_sync_results.txt 2>&1")
      
    ###########################################################################
    # Jobs that are complete and synced can be removed.
    ###########################################################################
    with open("wandb_sync_results.txt", "r") as sync_result_file:
        sync_result = sync_result_file.read()

    sync_success = sync_result.strip().endswith("done.")
    job_complete = os.path.exists(f"{f}/files/wandb-summary.json")
    
    if sync_success and job_complete:
        shutil.rmtree(f)
        tqdm.write(f"job {f} completed and synced successfully, removing... :)")
    elif sync_success and not job_complete:
        tqdm.write(f"job {f} synced successfully :)")
        pass    
    elif not sync_success:
        tqdm.write(f"job {f} encountered an error in syncing :( ")
        pass

os.remove("wandb_sync_results.txt")
