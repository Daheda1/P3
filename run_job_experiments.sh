#!/bin/bash
#SBATCH --job-name=U-net_Array
#SBATCH --output=result_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5  # Adjust based on the number of experiments

CONTAINER_PATH="/ceph/container/pytorch/pytorch_24.03.sif"

REQUIREMENTS_PATH="requirements.txt"
TRAIN_SCRIPT="train.py"

EXPERIMENTS=(
    "FullDatasetConfig"
    "MultiLossConfig"
    "FullDatasetConfig"
    "AlternateLossConfig"
    "LowerLearningRate"
    "HigherLearningRate"
)

CURRENT_EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Starting experiment $((SLURM_ARRAY_TASK_ID + 1))/12: $CURRENT_EXPERIMENT"

singularity exec "$CONTAINER_PATH" bash -c "
    pip install --user -r $REQUIREMENTS_PATH &&
    python $TRAIN_SCRIPT --config $CURRENT_EXPERIMENT
"

if [ $? -eq 0 ]; then
    echo "Experiment $CURRENT_EXPERIMENT completed successfully."
else
    echo "Experiment $CURRENT_EXPERIMENT failed. Check the error logs."
    exit 1
fi