#!/bin/bash
#SBATCH --job-name=EvenSmallerModel
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Number of GPUs to allocate (adjust this value as needed)
num_gpus=1

# Set the number of tasks and GPUs accordingly
#SBATCH --ntasks=$num_gpus
#SBATCH --gres=gpu:$num_gpus

singularity exec /ceph/container/pytorch/pytorch_24.03.sif bash -c "pip install --user -r requirements.txt && python train.py --config EvenSmallerModel"