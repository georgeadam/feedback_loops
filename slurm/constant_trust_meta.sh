#!/bin/bash

datasets=(mimic_iv_demographic)
models=(rf xgboost)
model_fprs=(0.1 0.2)
clinician_fprs=(0.01 0.05 0.1 0.2)
count=0

for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for model_fpr in ${model_fprs[@]}
        do
            for clinician_fpr in ${clinician_fprs[@]}
            do
                sbatch --wait --export=dataset=$dataset,model=$model,model_fpr=$model_fpr,clinician_fpr=$clinician_fpr constant_trust_skeleton.sh &
                count=$(( count + 1 ))

                if [[ ${count} -gt 7 ]]
                then
                    count=0
                    wait
                fi
            done
        done
    done
done