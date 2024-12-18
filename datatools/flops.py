import torch
from torch.profiler import profile, ProfilerActivity
from ptflops import get_model_complexity_info
import torch.nn as nn


# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
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
        self.activation = nn.Sigmoid()
    
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
        out = self.activation(out)
        return out

# Initialize the model and calculate FLOPs
model = UNet()
input_size = (3, 640, 640)  # 3 channels, 640x640 image

with torch.no_grad():
    macs, params = get_model_complexity_info(model, input_size, as_strings=False, verbose=False)

# Convert MACs to FLOPs (1 MAC = 2 FLOPs)
flops = 2 * macs
params, flops