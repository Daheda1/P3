# train.py

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import random
import numpy as np
import cv2 as cv
import os

# Import your existing modules
from modelparts.loss import calculate_loss
from modelparts.loadData import ExDark
from modelparts.modelStructure import UNet
from modelparts.imagePreprocessing import scale_image, scale_bounding_boxes, pad_image_to_target

# Checkpointing imports
import glob

class ExDarkDataset(Dataset):
    def __init__(self, dataset, image_paths, target_size=(1024, 1024), transform=None, divideable_by=32):
        self.dataset = dataset
        self.image_paths = image_paths
        self.target_size = target_size  # (width, height)
        self.transform = transform
        self.divideable_by = divideable_by

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image = self.dataset.load_image(image_name)
        ground_truth = self.dataset.load_ground_truth(image_name)

        image, resize_ratio = scale_image(image, self.target_size, self.divideable_by)
        image, padding = pad_image_to_target(image, self.target_size)
        ground_truth = scale_bounding_boxes(ground_truth, resize_ratio, padding, divideable_by=0)

        # Ensure the final image size is as expected
        assert image.size == (self.target_size[0], self.target_size[1]), (
            f"Image size after padding is {image.size}, expected {self.target_size}"
        )

        # Convert image to tensor
        image_tensor = TF.to_tensor(image)  # Automatically scales to [0,1]

        sample = {
            'image': image_tensor,
            'Org_image': image_tensor,
            'ground_truth': ground_truth,
            'padding': padding,
            'image_name': image_name  # Maybe not needed
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def custom_collate_fn(batch):
    batch_element = batch[0]
    collated_batch = {}
    for key in batch_element:
        if key in ['padding', 'ground_truth', 'image_name']:
            # Keep these as lists without collating into tensors
            collated_batch[key] = [d[key] for d in batch]
        else:
            # Use default collation for other fields
            collated_batch[key] = torch.stack([d[key] for d in batch], dim=0)
    return collated_batch

def train_model(model, dataloader, optimizer, num_epochs=25, device='cpu', checkpoint_dir='checkpoints'):
    print("Starting training...")
    model.train()

    # Initialize checkpointing
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if checkpoint_files:
        # Find the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            ground_truth = batch['ground_truth']
            padding = batch['padding']
            org_image = batch['Org_image']
            optimizer.zero_grad()
            output = model(image)
            loss = calculate_loss(output, org_image, ground_truth)
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {average_loss:.4f}")

        save_epoch_outputs(model, dataloader.dataset, epoch, device)

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


def save_epoch_outputs(model, dataset, epoch, device, output_dir="epoch_output_3"):
    """Save outputs of 5 random images and model checkpoint for the epoch."""
    # Create directory for the epoch
    epoch_dir = os.path.join(output_dir, f"Epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Select 5 random images
    random_images = random.sample(range(len(dataset)), 5)
    for idx in random_images:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
        image_name = sample['image_name']

        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            output = model(image)
            model.train()  # Reset model to training mode

        # Save input and output images
        input_image_path = os.path.join(epoch_dir, f"{os.path.splitext(image_name)[0]}_input.png")
        output_image_path = os.path.join(epoch_dir, f"{os.path.splitext(image_name)[0]}_output.png")

        # Save input and output
        save_image(image.squeeze(0), input_image_path)
        save_image(output.squeeze(0).cpu(), output_image_path)

    # Save the model checkpoint
    model_path = os.path.join(epoch_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    # Initialize dataset
    dataset_path = "DatasetExDark"
    dataset = ExDark(filepath=dataset_path)
    image_paths = dataset.load_image_paths_and_classes(split_filter=[1], class_filter=[1,2,3])  # Adjust filters as needed
    exdark_dataset = ExDarkDataset(dataset, image_paths, target_size=(1024, 1024))

    # Create data loader
    dataloader = DataLoader(
        exdark_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        collate_fn=custom_collate_fn
    )

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Start training with checkpointing
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=15,
        device=device,
        checkpoint_dir='checkpoint_3'  # Directory to save checkpoints
    )