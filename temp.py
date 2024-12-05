# test_validation.py

import torch
from torch import nn
from modelparts.loadData import ExDark, ExDarkDataset, custom_collate_fn
from modelparts.loss import calculate_loss
from modelparts.validation import validate_epoch
from modelparts.modelStructure import UNet
from torch.utils.data import DataLoader

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
    #print("Checkpoint keys:", checkpoint.keys())
    
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

def test_validation():
    # Set device to CPU
    device = torch.device('cpu')

    # Load the trained model directly
    model_path = 'Epoch13/model_epoch_13.pth'
    model = load_trained_model(model_path, device)

    # Load the ExDark dataset
    dataset_path = 'DatasetExDark'  # Replace with your actual dataset path
    dataset = ExDark(filepath=dataset_path)

    # Load validation image paths without slicing to use the entire validation set
    image_paths = dataset.load_image_paths_and_classes(split_filter=[2], class_filter=[2])[:3]

    # Create ExDarkDataset for validation
    target_size = (640, 640)
    validation_dataset = ExDarkDataset(dataset, image_paths, target_size)

    # Create DataLoader for validation
    batch_size = 32
    num_workers = 0
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # Ensure custom_collate_fn is imported or defined in this scope
    )

    # Define a simple loss function (e.g., Mean Squared Error)
    loss_function = calculate_loss

    # Dummy training loss for the epoch (since we're testing validation)
    train_loss = 0.0

    # Call the validate_epoch function
    validate_epoch(
        model=model,
        epoch=1,
        device=device,
        validation_loader=validation_loader,
        loss_function=loss_function,
        train_loss=train_loss,
        class_ids=[1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60],
        CSV_file="results.csv"
    )

if __name__ == "__main__":
    test_validation()