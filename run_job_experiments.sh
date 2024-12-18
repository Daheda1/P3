#!/bin/bash

# Define the array of experiments
EXPERIMENTS=(
    "FullDatasetConfig"
    

)

# Iterate over the experiments and submit each as a separate job
for i in "${!EXPERIMENTS[@]}"; do
    EXPERIMENT_NAME=${EXPERIMENTS[$i]}
    
    sbatch --job-name="$EXPERIMENT_NAME" <<EOF
#!/bin/bash
#SBATCH --output=result_${EXPERIMENT_NAME}_%A.out
#SBATCH --error=error_${EXPERIMENT_NAME}_%A.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Experiment-specific variables
CONTAINER_PATH="/ceph/container/pytorch/pytorch_24.03.sif"
REQUIREMENTS_PATH="requirements.txt"
TRAIN_SCRIPT="train.py"

echo "Starting experiment $EXPERIMENT_NAME"

singularity exec "\$CONTAINER_PATH" bash -c "
    pip install --user -r \$REQUIREMENTS_PATH &&
    python \$TRAIN_SCRIPT --config $EXPERIMENT_NAME
"

if [ \$? -eq 0 ]; then
    echo "Experiment $EXPERIMENT_NAME completed successfully."
else
    echo "Experiment $EXPERIMENT_NAME failed. Check the error logs."
    exit 1
fi
EOF

done