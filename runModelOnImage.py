# load_and_save_model_outputs.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from modelparts.modelStructure import UNet
from modelparts.loadData import ExDark, ExDarkDataset, custom_collate_fn

def load_trained_model(checkpoint_path, device):
    """
    Load the trained UNet model from a .pth checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    # Initialize the model architecture
    model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Debug: Print the keys in the checkpoint
    print("Checkpoint keys:", checkpoint.keys())
    
    # Attempt to load the state dictionary
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is the state_dict itself
        model.load_state_dict(checkpoint)
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully from {checkpoint_path}")
    return model

def initialize_dataloader(dataset_path, split_filter, class_filter, batch_size=32, num_workers=8, target_size=(640, 640)):
    """
    Initialize the DataLoader with specified filters.

    Args:
        dataset_path (str): Path to the dataset.
        split_filter (list): List indicating the data split to use.
        class_filter (list): List indicating the class filters.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 8.
        target_size (tuple, optional): Desired image size. Defaults to (640, 640).

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    # Initialize the dataset
    dataset = ExDark(filepath=dataset_path)
    
    # Load image paths with the specified filters
    image_paths = dataset.load_image_paths_and_classes(split_filter=split_filter, class_filter=class_filter)[0:3]
    
    # Create the ExDarkDataset
    exdark_dataset = ExDarkDataset(dataset, image_paths, target_size)
    
    # Create the DataLoader
    dataloader = DataLoader(
        exdark_dataset,
        batch_size=batch_size,
        shuffle=False,  # Typically False for evaluation
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    print(f"DataLoader initialized with split_filter={split_filter} and class_filter={class_filter}")
    return dataloader

def save_model_outputs(model, dataloader, device, output_dir="model_outputs"):
    """
    Perform inference on the DataLoader and save the output images to a specified folder.

    Args:
        model (torch.nn.Module): The trained model for inference.
        dataloader (DataLoader): DataLoader providing the data for inference.
        device (torch.device): Device to perform computations on.
        output_dir (str, optional): Directory to save the output images. Defaults to "model_outputs".
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving output images to '{output_dir}'")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            image_names = batch['image_name']  # Assuming 'image_name' is part of the dataset samples

            # Forward pass
            outputs = model(images)

            # Process and save each image in the batch
            for i in range(images.size(0)):
                input_image = images[i].cpu()
                output_image = outputs[i].cpu()
                image_name = image_names[i] if 'image_name' in batch else f"batch{batch_idx}_image{i}"

                # Define paths
                input_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_input.png")
                output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_output.png")

                # Save images
                save_image(input_image, input_image_path)
                save_image(output_image, output_image_path)

                print(f"Saved Input Image: {input_image_path}")
                print(f"Saved Output Image: {output_image_path}")

            print(f"Processed and saved batch {batch_idx + 1}/{len(dataloader)}")

    print("All output images have been saved successfully.")

def main():
    # Configuration Parameters
    checkpoint_path = "checkpoint_epoch_7.pth"  # Replace with your .pth file path
    dataset_path = "DatasetExDark"  # Replace with your dataset path
    split_filter = [3]  # Using split_filter=3 as per your request
    class_filter = [1]  # Assuming the same class_filter; adjust if needed
    batch_size = 32
    num_workers = 8
    target_size = (640, 640)
    output_dir = "model_outputs"  # Directory to save output images

    class Config:
        def __init__(self):
            self.class_filter = []
            self.light_filter = None
            self.location_filter = None
            self.target_size = (640, 640)
            self.batch_size = 32
            self.num_workers = 8
            self.class_ids = [1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]
            self.dataset_path = "DatasetExDark"

    config = Config()
    config.class_filter = [1, 2]

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model = load_trained_model(checkpoint_path, device)

    dataset = ExDark(filepath=config.dataset_path)
    image_paths = dataset.load_image_paths_and_classes(config, split_filter=[2])[:5]
    exdark_dataset = ExDarkDataset(dataset, image_paths, config.target_size)

    dataloader = DataLoader(
        exdark_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    # Perform inference and save outputs
    save_model_outputs(model, dataloader, device, output_dir=output_dir)

    print("Model loading, inference, and output saving complete.")

if __name__ == "__main__":
        main()