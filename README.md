# URSA: Unsupervised Latent Space Augmentation

### Setup
1. Install the following packages via the following commands.
    ```
    conda create -n py310URSA python=3.10 -y
    conda activate py310URSA
    conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge -y
    conda install -c conda-forge tqdm matplotlib gdown wandb -y
    pip install lmdb
    pip install git+https://github.com/tristanengst/apex-utils
    pip install git+https://github.com/rwightman/pytorch-image-models
    pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    ```
2. See **Setting up Imagenet** below for how to download Imagenet.
3. Download the ViT-Base checkpoint used in the original paper:
    ```
    python DownloadOriginalMAE.py --model vit_base
    ```

### Running Code
Finetune a ViT model using IMLE and training it as a VAE. This will save checkpoints including model weights and generate outputs to the `vaes` folder. You can also set `--resume` to `"none"` to train from scratch.
```
python FineTuneIMLEWithIMLE.py --resume vit_base
```
Run a linear evaluation using the encoder of a saved checkpoint as a frozen feature extractor:
```
foo
```
Finetune the encoder of a saved checkpoint on a classification task:
```
foo
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
