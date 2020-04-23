#!/bin/bash

datasets=(mimic_iv_demographic)
temporals=(temporal non_temporal)
models=(lr xgboost rf)
count=0

for dataset in ${datasets[@]}
do
    for temporal in ${temporals[@]}
    do
        for model in ${models[@]}
        do
            sbatch --wait --export=dataset=$dataset,temporal=$temporal,model=$model auc_importance_skeleton.sh &
            count=$(( count + 1 ))

            if [[ ${count} -gt 15 ]]
            then
                count=0
                wait
            fi
        done
    done
done