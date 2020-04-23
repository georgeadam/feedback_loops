#!/bin/bash
#SBATCH --mincpus=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=12G

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

script=src/scripts/compare_update_types.py

save_dir="figures/dynamic_threshold/${dataset}/${model}/${cal_partition}/${balance}/${update_type}/"
file_name=timestamp
rate_types=(auc)
temporal=True
train_year_limit=1997
update_year_limit=2019
seeds=1
threshold_validation_percentage=0.2
initial_desired_value=0.2
dynamic_desired_rate=fpr

python -u $script --data-type=$dataset --model=$model --initial-desired-value=$initial_desired_value --dynamic-desired-rate=$dynamic_desired_rate \
     --dynamic-desired-partition=$cal_partition --threshold-validation-percentage=$threshold_validation_percentage --update-types=$update_type \
     --save-dir=$save_dir --file-name=$file_name --rate-types "${rate_types[@]}" --balanced=$balance --temporal=$temporal \
      --train-year-limit=$train_year_limit --update-year-limit=$update_year_limit &

wait