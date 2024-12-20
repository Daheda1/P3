import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the directory containing the results
base_path = "results_to_download"  # Update if the path is incorrect

# List of files to include in the graph
files_to_use = [
    "lower_learning_rate_experiment.csv",
    "higher_learning_rate_experiment.csv"
]

# Initialize a dictionary to store data for plotting
experiment_data = {}

print("Processing specified files:")
# Loop through the specified files
for file in files_to_use:
    results_file = os.path.join(base_path, file)
    print(f"Checking file: {results_file}")
    if os.path.isfile(results_file):
        # Read the CSV file
        df = pd.read_csv(results_file)
        # Ensure the file has the necessary columns
        if 'Epoch' in df.columns and 'mAP' in df.columns:
            print(f"Loaded data from {results_file}:\n{df.head()}")
            # Extract the experiment name from the file name
            experiment_name = file.replace("_results.csv", "")
            # Store the mAP values with the corresponding epochs
            experiment_data[experiment_name] = df[['Epoch', 'mAP']]
        else:
            print(f"Skipping file (missing required columns): {results_file}")
    else:
        print(f"File not found: {results_file}")

# Baseline value
baseline = 0.10826223343610764

# Plotting
if experiment_data:
    plt.figure(figsize=(12, 6))
    for experiment, data in experiment_data.items():
        plt.plot(data['Epoch'], data['mAP'], label=experiment)

    # Add baseline as a red horizontal line
    plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline (0.1083)')

    # Add labels, title, legend
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP over Epochs for diffrent learning rates")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("No data found for plotting.")