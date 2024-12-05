# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import your existing modules
from modelparts.loss import calculate_loss
from modelparts.loadData import ExDark, ExDarkDataset, custom_collate_fn
from modelparts.modelStructure import UNet
from modelparts.validation import validate_epoch

# Checkpointing imports
import glob


from torchvision.utils import save_image
import random

def train_model(model, dataloader, validation_loader, optimizer, num_epochs, dataset, device, checkpoint_dir):
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

        #save_epoch_outputs(model, dataloader.dataset, epoch, device)
        #validate_epoch(validation_loader, model, device)

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

def save_epoch_outputs(model, dataset, epoch, device, output_dir="epoch_output_4"):
    """Save outputs of 5 random images and model checkpoint for the epoch."""
    # Create directory for the epoch
    epoch_dir = os.path.join(output_dir, f"Epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Select 5 random images
    random_images = random.sample(range(len(dataset)), 3)
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
    batch_size = 32
    num_workers = 8
    target_size = (640, 640)


    dataset = ExDark(filepath=dataset_path)
    image_paths = dataset.load_image_paths_and_classes(split_filter=[1], class_filter=[1])[:2]  # Adjust filters as needed
    exdark_dataset = ExDarkDataset(dataset, image_paths, target_size)

    # Create data loader
    dataloader = DataLoader(
        exdark_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    # Load validation image paths without slicing to use the entire validation set
    image_paths = dataset.load_image_paths_and_classes(split_filter=[2], class_filter=[1])[:2]

    # Create ExDarkDataset for validation
    validation_dataset = ExDarkDataset(dataset, image_paths, target_size)

    # Create DataLoader for validation
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # Ensure custom_collate_fn is imported or defined in this scope
    )

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Start training with checkpointing
    train_model(
        model=model,
        dataloader=dataloader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        num_epochs=15,
        dataset=dataset,
        device=device,
        checkpoint_dir='checkpoint_4'  # Directory to save checkpoints
    )




