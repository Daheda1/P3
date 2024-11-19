# train.py

import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import logging

# Import your existing modules
from modelparts.loss import calculate_loss
from modelparts.loadData import ExDark

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_dim, out_dim):
            block = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )
            return block
        
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.center = conv_block(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        
        center = self.center(pool4)
        
        up4 = self.up4(center)
        merge4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(merge4)
        
        up3 = self.up3(dec4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(merge3)
        
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(merge2)
        
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(merge1)
        
        out = self.final(dec1)
        return out

# Define the Dataset
class ExDarkDataset(Dataset):
    def __init__(self, dataset, image_paths, target_size=(512, 512), transform=None):
        self.dataset = dataset
        self.image_paths = image_paths
        self.target_size = target_size  # Desired (height, width)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_bgr = self.dataset.load_image(image_name)
        ground_truth = self.dataset.load_ground_truth(image_name)
        #print(f"Image: {image_name}, Ground Truth: {ground_truth}")
        
        if image_bgr is None:
            raise ValueError(f"Image {image_name} could not be loaded.")
        
        # Keep the original BGR image for calculate_loss
        image_cv2_bgr = image_bgr.copy()
        
        # Process the image for the model (convert to RGB and normalize)
        image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 255.0  # Normalize pixel values to [0, 1]
        image = image.permute(2, 0, 1)  # Rearrange dimensions to [C, H, W]
        
        original_size = image.shape[1:]  # Original (H, W)
        
        # Calculate padding
        padding = self.get_padding(original_size, self.target_size)
        padded_image = F.pad(image, padding, mode='constant', value=0)
        
        sample = {
            'image': padded_image,
            'image_cv2_bgr': image_cv2_bgr,
            'ground_truth': ground_truth,
            'original_size': original_size,
            'padding': padding,
            'image_name': image_name
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def get_padding(self, original_size, target_size):
        h_pad = target_size[0] - original_size[0]
        w_pad = target_size[1] - original_size[1]
        # Pad equally on both sides
        padding = (
            w_pad // 2, w_pad - w_pad // 2,  # Left, Right
            h_pad // 2, h_pad - h_pad // 2   # Top, Bottom
        )
        return padding

def custom_collate_fn(batch):
    batch_element = batch[0]
    collated_batch = {}
    for key in batch_element:
        if key in ['padding', 'ground_truth', 'original_size', 'image_name', 'image_cv2_bgr']:
            # Keep these as lists without collating into tensors
            collated_batch[key] = [d[key] for d in batch]
        else:
            # Use default collation for other fields
            collated_batch[key] = default_collate([d[key] for d in batch])
    return collated_batch

# Function to remove padding
def remove_padding(output, padding):
    # Ensure padding is a tuple of four integers
    if isinstance(padding, torch.Tensor):
        padding = padding.tolist()
    left, right, top, bottom = padding
    h_start = top
    h_end = output.shape[1] - bottom
    w_start = left
    w_end = output.shape[2] - right
    return output[:, h_start:h_end, w_start:w_end]

# Training loop
def train_model(model, dataloader, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            image_cv2_bgrs = batch['image_cv2_bgr']  # List of cv2 BGR images
            ground_truths = batch['ground_truth']  # List of ground truth annotations
            paddings = batch['padding']
            image_names = batch['image_name']
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Remove padding from outputs
            outputs_unpadded = []
            for i in range(outputs.size(0)):
                output = outputs[i]
                padding = paddings[i]  # This should be a tuple (left, right, top, bottom)
                output_unpadded = remove_padding(output, padding)
                outputs_unpadded.append(output_unpadded)
            # outputs_unpadded is a list of tensors with possibly different sizes
    
            batch_loss = 0.0
            for i in range(len(outputs_unpadded)):
                output_unpadded = outputs_unpadded[i]
                ground_truth = ground_truths[i]  # Ground truth for the current image
                image_cv2_bgr = image_cv2_bgrs[i]  # BGR image in cv2 format
                
                # Move output to CPU if necessary for calculate_loss
                output_unpadded = output_unpadded.cpu().detach().numpy()
                # Compute loss
                loss = calculate_loss(image_cv2_bgr, ground_truth)
                batch_loss += loss
    
            # Backpropagate and optimize
            batch_loss.backward()
            optimizer.step()
            
            running_loss += batch_loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    # Initialize dataset
    dataset_path = "DatasetExDark"
    dataset = ExDark(filepath=dataset_path)
    image_paths = dataset.load_image_paths_and_classes(split_filter=[2])  # Adjust filters as needed
    exdark_dataset = ExDarkDataset(dataset, image_paths, target_size=(512, 512))
    
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
    train_model(model, dataloader, optimizer, num_epochs=25)