# IMLE-SSL: Implicit Maximum Likelihood for Self-Supervised Learning

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

For the runs referenced in the graph, the IMLE-MAE run can be recreated with
```
TrainIMLE.py --gpus 0 1 --suffix IMLE_FROM_SCRATCH --data_tr IMAGENET_TRAIN --data_val IMAGENET_VAL --fast_linear_probe 1 --ex_for_mse_loss 2048 --ip_spec 11 adain --ex_per_epoch 2048 --code_bs 32 --sp 64 --ns 128 --mini_bs 64 --ipe 32 --lr_z 1e-4 --lr 1e-4 --use_augs 1 --shuffle_data 1 --scheduler linear_ramp_cosine_decay --n_ramp 8 --epochs 512 --save_iter 32 --train_n_way 32 --train_n_shot 64 --steps_per_eval 256 --wandb offline --finetune 0 --ignore_z 0 --no-fp16 --num_workers 8
```
and the MAE run with
```
TrainIMLE.py --gpus 0 1 --suffix MAE_FROM_SCRATCH --data_tr IMAGENET_TRAIN --data_val IMAGENET_VAL --fast_linear_probe 1 --ex_for_mse_loss 2048 --ip_spec 11 adain --ex_per_epoch 2048 --code_bs 32 --sp 64 --ns 128 --mini_bs 64 --ipe 32 --lr_z 1e-4 --lr 1e-4 --use_augs 1 --shuffle_data 1 --scheduler linear_ramp_cosine_decay --n_ramp 8 --epochs 512 --save_iter 32 --train_n_way 32 --train_n_shot 64 --steps_per_eval 256 --wandb offline --finetune 0 --ignore_z 1 --no-fp16 --num_workers 8
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
