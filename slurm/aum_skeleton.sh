#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

echo "Slurm node ID"
echo $SLURM_NODEID
echo "Slurm JOB ID"
echo $SLURM_JOBID
cd ..

PYTHONPATH=$(pwd)
export PYTHONPATH
export HYDRA_FULL_ERROR=1

source /h/alexadam/anaconda3/bin/activate dl
model=resnet34
augmentation=False
noise=0.2
weight_decay=0.000001

script=src/scripts/aum.py

python -u $script model.name=$model data.augmentation=$augmentation data.noise=$noise optim.weight_decay=$weight_decay &

wait