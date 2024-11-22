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

# Import your existing modules
from modelparts.loss import calculate_loss
from modelparts.loadData import ExDark
from modelparts.modelStructure import UNet
from modelparts.imagePreprocessing import scale_image, scale_bounding_boxes, pad_image_to_target

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
        ground_truth = scale_bounding_boxes(ground_truth, resize_ratio, padding, divideable_by=self.divideable_by)


        # Ensure the final image size is as expected
        assert image.size == (self.target_size[0], self.target_size[1]), (
            f"Image size after padding is {image.size}, expected {self.target_size}"
        )

        # Convert image to tensor
        image_tensor = TF.to_tensor(image)  # Automatically scales to [0,1]

        sample = {
            'image': image_tensor,
            'ground_truth': ground_truth,
            'padding': padding,
            'image_name': image_name #Maybe not needed
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


def train_model(model, dataloader, optimizer, num_epochs=25):
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            image = batch['image']
            ground_truth = batch['ground_truth']
            padding = batch['padding']
            image = image.to(device)
            ground_truth = batch['ground_truth']
            optimizer.zero_grad()
            output = model(image)
            loss = calculate_loss(output, ground_truth)
            print(f"LOSS: {loss}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")



if __name__ == "__main__":
    # Initialize dataset
    dataset_path = "DatasetExDark"
    dataset = ExDark(filepath=dataset_path)
    image_paths = dataset.load_image_paths_and_classes(split_filter=[2], class_filter=[1])[:2]  # Adjust filters as needed
    exdark_dataset = ExDarkDataset(dataset, image_paths, target_size=(1024, 1024))
    
    # Create data loader
    dataloader = DataLoader(
        exdark_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )    
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Start training
    train_model(model, dataloader, optimizer, num_epochs=15)