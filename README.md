# 3MRD: Masked Multi-Modal Representation Densities

### Setup
Install the following packages via the following commands.
```
conda create -n py3103MVR python=3.10
conda activate py3103MVR
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge -y
conda install -c conda-forge tqdm matplotlib gdown wandb -y
pip install lmdb
pip install git+https://github.com/tristanengst/apex-utils
pip install git+https://github.com/rwightman/pytorch-image-models
pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
```
Download the datasets we use:
```
python DownloadData.py --datasets miniImagenet
```
Download the ViT-Base checkpoint used in the original paper:
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
