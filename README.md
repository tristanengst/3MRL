# URSA: Unsupervised Latent Space Augmentation

### Setup
1. Install the following packages via the following commands.
    ```
    conda create -n py310URSA python=3.10 -y
    conda activate py310URSA
    conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge -y
    pip install tqdm matplotlib gdown wandb lmdb
    pip install git+https://github.com/tristanengst/Misc
    pip install git+https://github.com/rwightman/pytorch-image-models
    pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    ```
2. Download the pretrained MAE weights from [here](https://github.com/facebookresearch/mae/issues/8).

### Running Code
Finetune a ViT model using IMLE and training it as a implicit probabilistic model. Please see `IO.py` for the arguments this takes.
```
python TrainIMLE.py --data_tr PATH_TO_TRAINING_DATA --data_val PATH_TO_VAL_DATA
```

### Downloadable Checkpoints

### Setting up ImageNet
Make sure you're working on a computer on which you can have _a lot_ of files. Some compute clusters like ComputeCanada don't allow this. Download [ImageNet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)—you need the task one and two training images and development kit, along with the validation images for all tasks. Put the files in `data/imagenet`, and run
```
python -c 'import torchvision; torchvision.datasets.ImageNet("data/imagenet", split="val")'
python -c 'import torchvision; torchvision.datasets.ImageNet("data/imagenet", split="train")'
```
If you need to run on a system with a limited number of files, run
```
tar -cvf data/imagenet/train.tar data/imagenet/train
tar -cvf data/imagenet/val.tar data/imagenet/val
```
and transfer the two resulting TAR files to the file-limited system. They can be used as drop-in replacements for the ImageFolder-compatible directories we normally use, eg. `--data_tr data/imagenet/train.tar`.
