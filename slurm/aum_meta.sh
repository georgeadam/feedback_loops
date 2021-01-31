#!/bin/bash

models=(resnet34)
augmentations=(True False)
noises=(0.1 0.2)
weight_decays=(0.0 0.000001 0.00001 0.0001)
count=0

for model in ${models[@]}
do
    for augmentation in ${augmentations[@]}
    do
        for noise in ${noises[@]}
        do
            for weight_decay in ${weight_decays[@]}
            do
                sbatch --wait --export=model=$model,augmentation=$augmentation,noise=$noise,weight_decay=$weight_decay aum_skeleton.sh &
                count=$(( count + 1 ))
                sleep 1

                if [[ ${count} -gt 15 ]]
                then
                    count=0
                    wait
                fi
            done
        done
    done
done