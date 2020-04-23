#!/bin/bash

datasets=(mimic_iv_demographic)
models=(rf xgboost)
count=0

for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        sbatch --wait --export=dataset=$dataset,model=$model regularization_skeleton.sh &
        count=$(( count + 1 ))

        if [[ ${count} -gt 15 ]]
        then
            count=0
            wait
        fi
    done
done