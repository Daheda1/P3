from modelparts.loss import calculate_loss
from modelparts.loadData import ExDark
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

base_folder1 = "DatasetExDark/ExDark_Annno"

def plot_class_loss_boxplot_and_scatter(loss_data):
    # Create a boxplot for loss distribution per class
    
    fig1, axs = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [1, 4]})

    # Boxplot
    axs[0].boxplot([losses for losses in loss_data.values() if len(losses) > 0], vert=True, patch_artist=True)
    axs[0].set_xticks(range(1, len(loss_data) + 1))
    axs[0].set_xticklabels(loss_data.keys(), rotation=45)
    axs[0].set_title("Loss Distribution per Class")
    axs[0].set_xlabel("Class")
    axs[0].set_ylabel("Loss")
    axs[0].set_ylim(0, 3)

    # Scatter plots for each class
    num_classes = len(loss_data)
    num_cols = 5
    num_rows = (num_classes + num_cols - 1) // num_cols  # Calculate rows dynamically

    fig2, scatter_figs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    fig2.suptitle("Loss Scatter Plot per Class")

    # Flatten axis array for easier indexing
    scatter_figs = scatter_figs.flatten() if num_rows > 1 else [scatter_figs]

    for i, (class_name, losses) in enumerate(loss_data.items()):
        if len(losses) == 0:
            continue  # Skip empty classes
        scatter_figs[i].scatter(range(len(losses)), losses)
        scatter_figs[i].set_title(class_name)
        scatter_figs[i].set_xlabel("Image Index")
        scatter_figs[i].set_ylabel("Loss")
        scatter_figs[i].set_ylim(0, 3)

    # Hide unused subplots
    for j in range(i + 1, len(scatter_figs)):
        scatter_figs[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset = ExDark(filepath="DatasetExDark")
    class_stats = {}
    loss_data = {}  # New dictionary to store raw loss values for each class

    for i in tqdm(range(12), desc="Processing Classes"):  # Progress bar for class processing
        filepaths = dataset.load_image_paths_and_classes(split_filter=[2], class_filter=[i+1])
        loss_list = []
        
        for filepath in tqdm(filepaths, desc=f"Processing Images for Class {i+1}", leave=False):  # Nested progress bar
            image = dataset.load_image(filepath)
            ground_truth = dataset.load_ground_truth(filepath)
            classes = [1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]

            loss = calculate_loss(image, ground_truth, classes)
            loss_list.append(loss)
        
        # Store raw loss values for each class in loss_data
        loss_data[f'class_{i+1}'] = loss_list

        # Convert to numpy array for statistical calculations
        losses = np.array(loss_list)
        class_stats[f'class_{i+1}'] = {
            'Mean': np.mean(losses),
            'Median': np.median(losses),
            'High': np.max(losses),
            'Low': np.min(losses),
            '25%': np.percentile(losses, 25),
            '75%': np.percentile(losses, 75)
        }

    # Convert to DataFrame and export to CSV
    stats_df = pd.DataFrame(class_stats).T
    stats_df.to_csv("Class_Loss_Statistics.csv")
    print("Class Loss Statistics exported to 'Class_Loss_Statistics.csv'")
    plot_class_loss_boxplot_and_scatter(loss_data)