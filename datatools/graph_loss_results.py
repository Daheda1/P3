import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the directory containing the results
base_path = "results"  # Update if the path is incorrect

# List of files to include in the graph
files_to_use = [
    "base_object_light_experiment_results.csv"
]

# Function to plot Train Loss and Eval Loss for a single experiment
def plot_losses(file_path, experiment_name):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}:\n{df.head()}")

        # Check for necessary columns
        required_columns = ['Epoch', 'Train Loss', 'Eval Loss']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping file (missing required columns): {file_path}")
            return

        # Divide Eval Loss by 32
        df['Eval Loss'] = df['Eval Loss'] / 32
        print(f"Adjusted Eval Loss by dividing by 32:\n{df[['Epoch', 'Eval Loss']].head()}")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
        plt.plot(df['Epoch'], df['Eval Loss'], label='Eval Loss', marker='s')

        # Add labels, title, legend, and grid
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Train vs Eval Loss for {experiment_name}")
        plt.legend()
        plt.grid(True)

        # Improve layout and show the plot
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

def main():
    print("Processing specified files:")
    # Loop through the specified files
    for file in files_to_use:
        results_file = os.path.join(base_path, file)
        print(f"\nChecking file: {results_file}")
        if os.path.isfile(results_file):
            # Extract the experiment name from the file name
            experiment_name = os.path.splitext(file)[0]  # Removes the .csv extension
            # Plot Train Loss and Eval Loss for this experiment
            plot_losses(results_file, experiment_name)
        else:
            print(f"File not found: {results_file}")

if __name__ == "__main__":
    main()