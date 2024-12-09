# train.py

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os

# Import your existing modules
from modelparts.loss import calculate_loss
from modelparts.loadData import ExDark, ExDarkDataset, custom_collate_fn
from modelparts.modelStructure import UNet
from modelparts.validation import validate_epoch
from modelparts.yolo import init_yolo

# Checkpointing imports
import glob
import logging

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

import argparse
import inspect
import sys
import configs


# Dynamically load all configuration classes from configs.py
def load_configs(module):
    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__  # Ensure classes are from the configs module
    }
image_loss_fn = nn.L1Loss()

def train_model(model, yolo_model, dataloader, validation_loader, optimizer, config):
    print("Starting training...")
    model.train()

    # Initialize checkpointing
    os.makedirs(config.experiment_name, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(config.experiment_name, 'checkpoint_epoch_*.pth'))
    if checkpoint_files:
        # Find the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, config.num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            image = batch['image'].to(config.device)
            original_image = batch['Org_image']
            ground_truth = batch['ground_truth']
            optimizer.zero_grad()
            output = model(image)
            if epoch in config.alt_loss_pattern:
                loss = image_loss_fn(output, original_image)
            else:
                loss = calculate_loss(output, ground_truth, yolo_model)
            print(f"Epoch [{epoch+1}/{config.num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {average_loss:.4f}")

        validate_epoch(model, yolo_model, epoch, validation_loader, calculate_loss, average_loss, config)

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(config.experiment_name, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")



if __name__ == "__main__":
    CONFIG_MAP = load_configs(configs)
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a UNet model on ExDark dataset with configurable settings")
    parser.add_argument("--config", type=str, default="base", choices=CONFIG_MAP.keys(), help="Configuration to use")
    args = parser.parse_args()

    # Load the selected configuration
    config_class = CONFIG_MAP[args.config]
    config = config_class()

    # Initialize dataset
    class_ids = [1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]
    config.class_ids = [class_ids[i - 1] for i in config.class_filter]

    yolo_model = init_yolo(class_ids)

    dataset = ExDark(filepath=config.dataset_path)
    image_paths = dataset.load_image_paths_and_classes(config, split_filter=[1])
    exdark_dataset = ExDarkDataset(dataset, image_paths, config.target_size)

    dataloader = DataLoader(
        exdark_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    validation_image_paths = dataset.load_image_paths_and_classes(config, split_filter=[2])
    validation_dataset = ExDarkDataset(dataset, validation_image_paths, config.target_size)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    model = UNet(in_channels=3, out_channels=3).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_model(
        model=model,
        yolo_model=yolo_model,
        dataloader=dataloader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        config=config
    )