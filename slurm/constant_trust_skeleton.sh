#!/bin/bash
#SBATCH --mincpus=6
#SBATCH --partition=cpu
#SBATCH --mem=12G

cd ..

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

script=src/scripts/constant_trust_effect.py

save_dir="results/trust_experiments/constant/${dataset}/non_temporal/${model}/${clinician_fpr}/${model_fpr}/"
temporal=temporal
seeds=10
train_year_limit=1997
update_year_limit=2019


python -u $script data=$temporal data.type=$dataset model=$model rates.clinician_fpr=$clinician_fpr rates.model_fpr=$model_fpr  \
     data.tyl=$train_year_limit data.uyl=$update_year_limit misc.seeds=$seeds  &

wait