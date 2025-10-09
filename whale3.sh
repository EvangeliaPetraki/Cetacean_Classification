#!/bin/bash
#SBATCH --job-name=whale_No_aug
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere
#SBATCH --time=6:00:00


source ~/myenv/bin/activate

python3 ~/E/Cetacean_Classification/whale3.py
