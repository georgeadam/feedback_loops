#!/bin/bash

datasets=(mimic_iv mimic_iv_12h)
balances=(False True)
models=(lr random_forest xgboost)
partitions=(train update_cumulative update_current all)
update_types=(partial_confidence drop oracle)
count=0

for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for cal_partition in ${partitions[@]}
        do
            for update_type in ${update_types[@]}
            do
                for balance in ${balances[@]}
                do
                    sbatch --wait --export=dataset=$dataset,model=$model,cal_partition=$cal_partition,update_type=$update_type,balance=$balance slurm_dynamic_threshold_skeleton.sh &
                    count=$(( count + 1 ))

                    if [[ ${count} -gt 15 ]]
                    then
                        count=0
                        wait
                    fi
                done
            done
        done
    done
done