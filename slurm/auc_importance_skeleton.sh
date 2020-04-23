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

save_dir="results/paper/auc_importance/${dataset}/${model}/${temporal}/"
update_types="[ feedback_full_fit ]"
idr=fpr
idv=0.2
ddr=fpr
ddp=all
sorted=False
worst_case=False
clinician_fpr=0.0
warm_start=False
seeds=10
train_year_limit=1997
update_year_limit=2019
next_year=True
balanced=False
num_updates=50
use_cv=False

if [ $temporal = "temporal" ]
then
    python -u $script data=$temporal data.type=$dataset model=$model model.use_cv=$use_cv misc.seeds=$seeds \
         misc.update_types="${update_types}" rates.idr=$idr rates.idv=$idv rates.ddr=$ddr rates.ddp=$ddp data.next_year=$next_year \
         rates.clinician_fpr=$clinician_fpr model.warm_start=$warm_start data.tyl=$train_year_limit \
         data.uyl=$update_year_limit data.balanced=$balanced hydra.run.dir=$save_dir &
else
    python -u $script data=$temporal data.type=$dataset model=$model model.use_cv=$use_cv misc.seeds=$seeds \
         misc.update_types="${update_types}" rates.idr=$idr rates.idv=$idv rates.ddr=$ddr rates.ddp=$ddp data.next_year=$next_year \
         data.worst_case=$worst_case rates.clinician_fpr=$clinician_fpr model.warm_start=$warm_start \
         data.num_updates=$num_updates data.balanced=$balanced hydra.run.dir=$save_dir &
fi



wait