#!/bin/bash

datasets=(mimic_iv mimic_iv_demographic mimic_iv_12h mimic_iv_12h_demographic)
count=0

for dataset in ${datasets[@]}
do
    sbatch --wait --export=dataset=$dataset update_types_skeleton.sh &
    count=$(( count + 1 ))

    if [[ ${count} -gt 15 ]]
    then
        count=0
        wait
    fi
done