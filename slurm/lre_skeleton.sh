#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --partition=cpu
#SBATCH --mem=8G

echo "Slurm node ID"
echo $SLURM_NODEID
echo "Slurm JOB ID"
echo $SLURM_JOBID
cd ..

PYTHONPATH=$(pwd)
export PYTHONPATH
export HYDRA_FULL_ERROR=1

source /h/alexadam/anaconda3/bin/activate dl

balanced=True

script=src/scripts/lre.py

python -u $script data.type=$dataset data.noise=$noise data.n_update=$n_update data.n_train=$n_train data.n_test=$n_test data.balanced=$balanced &

wait