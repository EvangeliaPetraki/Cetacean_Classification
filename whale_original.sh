#!/bin/bash
#SBATCH --job-name=whale_test
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere
#SBATCH --time=1:00:00


source ~/myenv/bin/activate

python3 ~/E/Cetacean_Classification/whale_original.py
