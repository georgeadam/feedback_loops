#!/bin/bash
#SBATCH --mincpus=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

cd ..

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

script=src/scripts/compare_update_types.py

save_dir="results/paper/update_types_new/${dataset}/non_temporal/"
temporal=non_temporal
update_types="[ feedback_online_single_batch, feedback_online_all_update_data, feedback_online_all_data, feedback_full_fit_cad, feedback_full_fit_past_year_cad, evaluate ]"
idr=fpr
idv=0.2
ddr=fpr
ddp=all
clinician_fpr=0.0
seeds=10
model=nn
lr=0.0001
online_lr=0.0001
optimizer=adam
reset_optim=False
iterations=3000
tol=0.0001
hidden_layers=0
soft=False
train_year_limit=1997
update_year_limit=2019

python -u $script data=$temporal data.type=$dataset model=$model misc.seeds=$seeds misc.update_types="${update_types}" \
     rates.idr=$idr rates.idv=$idv rates.ddr=$ddr rates.ddp=$ddp rates.clinician_fpr=$clinician_fpr model.lr=$lr \
     model.online_lr=$online_lr model.optimizer=$optimizer model.reset_optim=$reset_optim model.iterations=$iterations model.tol=$tol \
     model.hidden_layers=$hidden_layers data.tyl=$train_year_limit data.uyl=$update_year_limit model.soft=$soft &

wait