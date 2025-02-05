#!/bin/bash
#SBATCH --gpus 1
#SBATCH --job-name st
#SBATCH --nodelist ac01
#SBATCH --time 4-00:00:00
#SBATCH --output ./slurm_out/%x_%j.out
#SBATCH -c 8
#SBATCH --mem 100G

conda activate fact


####241209

python sequential_tuning2.py --epochs 10 --batch_size 1 --learning_rate 0.001

python sequential_tuning2.py --epochs 5 --batch_size 30 --learning_rate 0.05



######

# python sequential_tuning.py --epochs 3 --batch_size 30 --learning_rate 0.001

# python sequential_tuning.py --epochs 3 --batch_size 30 --learning_rate 0.005

# python sequential_tuning.py --epochs 10 --batch_size 1 --learning_rate 0.001 --eval_on_train

# python sequential_tuning.py --epochs 10 --batch_size 1 --learning_rate 0.005

# python sequential_tuning.py --epochs 5 --batch_size 15 --learning_rate 0.001

# python sequential_tuning.py --epochs 5 --batch_size 15 --learning_rate 0.005

##########
# python sequential_tuning_freeze.py --epochs 3 --batch_size 30 --learning_rate 0.001

# python sequential_tuning_freeze.py --epochs 3 --batch_size 30 --learning_rate 0.005

# python sequential_tuning_freeze.py --epochs 10 --batch_size 1 --learning_rate 0.001 --eval_on_train

# python sequential_tuning_freeze.py --epochs 10 --batch_size 1 --learning_rate 0.005

# python sequential_tuning_freeze.py --epochs 5 --batch_size 15 --learning_rate 0.001

# python sequential_tuning_freeze.py --epochs 5 --batch_size 15 --learning_rate 0.005

#######
# python sequential_tuning.py --epochs 3 --batch_size 30 --learning_rate 0.01

# python sequential_tuning.py --epochs 3 --batch_size 30 --learning_rate 0.05 --eval_on_train

# python sequential_tuning_freeze.py --epochs 3 --batch_size 30 --learning_rate 0.01

# python sequential_tuning_freeze.py --epochs 3 --batch_size 30 --learning_rate 0.05 --eval_on_train
####




# python sequential_tuning_lora.py --epochs 10 --batch_size 1 --learning_rate 0.001 --eval_on_train

# python sequential_tuning_lora.py --epochs 10 --batch_size 1 --learning_rate 0.005


# python sequential_tuning_lora.py --epochs 3 --batch_size 30 --learning_rate 0.01

# python sequential_tuning_lora.py --epochs 3 --batch_size 30 --learning_rate 0.05 --eval_on_train