#!/bin/bash

dataset=sklearn
noises=(0.1 0.2)
n_trains=(25 50 75 100)
n_updates=(100 200 500)
count=0

for n_train in ${n_trains[@]}
do
    for n_update in ${n_updates[@]}
    do
        for noise in ${noises[@]}
        do
            sbatch --wait --export=dataset=$dataset,n_train=$n_train,n_update=$n_update,noise=$noise lre_skeleton.sh &
            count=$(( count + 1 ))

            if [[ ${count} -gt 15 ]]
            then
                count=0
                wait
            fi

            sleep 2
        done
    done
done