#!/bin/bash
#SBATCH --mincpus=6
#SBATCH --partition=cpu
#SBATCH --mem=12G

cd ..

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

script=src/scripts/conditional_trust_effect.py

save_dir="results/trust_experiments/conditional/${dataset}/temporal/${model}/"
temporal=temporal
seeds=10
clinician_fprs="[ 0.01, 0.05, 0.1, 0.2 ]"
model_fprs="[ 0.2 ]"
train_year_limit=1997
update_year_limit=2019

python -u $script data=$temporal data.type=$dataset model=$model rates.clinician_fprs="${clinician_fprs}" rates.model_fprs="${model_fprs}"  \
     data.tyl=$train_year_limit data.uyl=$update_year_limit misc.seeds=$seeds hydra.run.dir=$save_dir &

wait