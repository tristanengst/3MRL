"""File for preparing the iMaterialist dataset.

To download the dataset, install the Kaggle python package, and run

kaggle competitions download -c imaterialist-challenge-fashion-2018

inside a folder. Then run this script.
"""

import argparse
import json
import os
import requests
from tqdm import tqdm
import zipfile

def prepare_data_file(d, args):
    """Creates an ImageFolder
    """
    with zipfile.ZipFile(d, "r") as f:
        f.extractall(os.path.dirname(d))

    with open(d.replace(".zip", ""), "r") as f:
        data = json.load(f)

    folder = os.path.dirname(d)
    split = os.path.basename(d).replace(".json.zip", "")
    if not os.path.exists(f"{folder}/{split}"):
        os.makedirs(f"{folder}/{split}")

    image_id2labels = {anno["imageId"]: anno["labelID"]
        for anno in data["annotations"]}

    for image_data in tqdm(data["images"],
        desc="Iterating over splits",
        dynamic_ncols=True):

        image_name = f"{folder}/{split}/{image_data['imageId']}.jpg"

        if args.avoid_download_if_exists and os.path.exists(image_name):
            pass
        else:
            image = requests.get(image_data["url"]).content
            with open(image_name, "wb") as f:
                f.write(image)

        image_name2labels[image_name] = image_id2labels[image_data["imageId"]]

    
    with open(f"{folder}/{split}/file2label.json", "wb") as f:
        json.dump(image_name2labels)

P = argparse.ArgumentParser()
P.add_argument("--imat_path", required=True,
    help="Path to folder containing iMaterialist dataset")
P.add_argument("--avoid_download_if_exists", choices=[0, 1], default=0,
    help="Whether to download already downloaded images")
args = P.parse_args()

data_files = [f"{args.imat_path}/{f}" for f in os.listdir(args.imat_path)
    if f.endswith(".json.zip") and "val" in f]

for d in tqdm(data_files,
    desc="Iterating over splits",
    dynamic_ncols=True):

    prepare_data_file(d, args)

