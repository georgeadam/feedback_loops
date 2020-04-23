#!/bin/bash
#SBATCH --mincpus=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=12G

cd ..

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

script=src/scripts/compare_update_types.py

save_dir="results/regularization/${dataset}/temporal/${model}/"
temporal=temporal
update_types="[ feedback_full_fit_oracle, feedback_full_fit_drop_random, feedback_full_fit_drop_everything, feedback_full_fit_drop_low_confidence, feedback_full_fit, no_feedback_full_fit, evaluate ]"
idr=fpr
idv=0.05
ddr=fpr
ddp=all
clinician_fpr=0.0
seeds=10
train_year_limit=1997
update_year_limit=2019

python -u $script data=$temporal data.type=$dataset model=$model misc.seeds=$seeds misc.update_types="${update_types}" \
     rates.idr=$idr rates.idv=$idv rates.ddr=$ddr rates.ddp=$ddp rates.clinician_fpr=$clinician_fpr data.tyl=$train_year_limit \
     data.uyl=$update_year_limit hydra.run.dir=$save_dir &

wait