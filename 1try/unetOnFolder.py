import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join, splitext, basename
import numpy as np
from tqdm import tqdm

# ---------------------------
# 1. U-Net Model Definition
# ---------------------------

class DoubleConv(nn.Module):
    """Applies two consecutive convolutional layers with Batch Normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Optional but recommended
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Optional but recommended
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool followed by DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),  # Use ceil_mode to handle odd dimensions
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling followed by DoubleConv."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # If bilinear interpolation is used for upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # If transposed convolution is used for upsampling
            self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Adjust padding to match the size of x2 (the corresponding feature map from the contracting path)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final convolution layer to map features to the desired number of output channels."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """The U-Net architecture."""
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # Number of input channels (e.g., 3 for RGB images)
        self.n_classes = n_classes    # Number of output channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)  # Initial convolution
        self.down1 = Down(64, 128)             # Downsampling steps
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1          # Adjust the number of channels if using bilinear upsampling
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)  # Upsampling steps
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)     # Final output layer

    def forward(self, x):
        x1 = self.inc(x)     # Encoder pathway
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)  # Decoder pathway with skip connections
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # Output layer

        # Ensure the output has the same spatial dimensions as the input
        diffY = x.size()[2] - logits.size()[2]
        diffX = x.size()[3] - logits.size()[3]

        logits = F.pad(logits, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        return logits

# ---------------------------
# 2. Utility Functions
# ---------------------------

def get_device():
    """Returns the available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_folder(model, input_folder, output_folder, device, mask_size=(512, 512)):
    """
    Applies the model to all images in input_folder and saves the predicted masks in output_folder.
    Args:
        model (nn.Module): Trained U-Net model.
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save the output masks.
        device (torch.device): Device to perform computation on.
        mask_size (tuple): Desired size of the mask (height, width).
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Define acceptable image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Get all image files in the input directory
    image_files = [
        f for f in listdir(input_folder)
        if isfile(join(input_folder, f)) and splitext(f)[1].lower() in image_extensions
    ]
    full_paths = [join(input_folder, f) for f in image_files]

    model.eval()

    with torch.no_grad():
        for image_path in tqdm(full_paths, desc="Processing images"):
            image_name = splitext(basename(image_path))[0]

            # Read and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image_rgb.shape[:2]

            # Resize image
            image_resized = cv2.resize(image_rgb, (mask_size[1], mask_size[0]))
            image_normalized = image_resized / 255.0
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)

            # Inference
            output = model(image_tensor)
            predicted_mask = output.squeeze().cpu().numpy()

            # Post-process predicted mask (e.g., thresholding)
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

            # Resize mask back to original size
            predicted_mask_resized = cv2.resize(predicted_mask, (original_size[1], original_size[0]))

            # Save the mask
            mask_save_path = join(output_folder, f"{image_name}_mask.png")
            cv2.imwrite(mask_save_path, predicted_mask_resized)

# ---------------------------
# 3. Main Script
# ---------------------------

if __name__ == "__main__":
    # Paths to the trained model checkpoint, input folder, and output folder
    checkpoint_path = "results/best_unet_checkpoint.pth.tar"
    input_folder = "DatasetExDark/ExDark_images"  # Replace with your input folder path
    output_folder = "results2"  # Replace with your output folder path

    # Device configuration
    device = get_device()
    print(f"Using device: {device}")

    # Instantiate the model
    n_channels = 3      # Number of input channels (RGB)
    n_classes = 1       # Single channel mask
    model = UNet(n_channels, n_classes)
    model = model.to(device)

    # Load the model state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # If using DataParallel, adjust the keys in state_dict
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` if present
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Apply the model to the entire folder
    predict_folder(model, input_folder, output_folder, device)
