#!/bin/bash

datasets=(mimic_iv_demographic)
models=(rf xgboost)
seeds_vals=(1 10)
count=0

for seeds in ${seeds_vals[@]}
do
    for model in ${models[@]}
    do
        for dataset in ${datasets[@]}
        do
            sbatch --wait --export=dataset=$dataset,model=$model,seeds=$seeds figure_1_skeleton.sh &
            count=$(( count + 1 ))

            if [[ ${count} -gt 15 ]]
            then
                count=0
                wait
            fi
        done
    done
done