import os
import time
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.ops import box_iou

from ultralytics import YOLO

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        def conv_block(in_feat, out_feat, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size, padding=padding),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_feat, out_feat, kernel_size, padding=padding),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        up4 = self.up4(bn)
        merge4 = torch.cat([up4, d4], dim=1)
        c4 = self.conv4(merge4)
        up3 = self.up3(c4)
        merge3 = torch.cat([up3, d3], dim=1)
        c3 = self.conv3(merge3)
        up2 = self.up2(c3)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)
        up1 = self.up1(c2)
        merge1 = torch.cat([up1, d1], dim=1)
        c1 = self.conv1(merge1)

        output = self.final(c1)
        return output

# Custom Dataset for ExDark
class ExDarkDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_paths = []
        valid_extensions = ['.png', '.jpg', '.jpeg']
        for f in os.listdir(image_dir):
            if os.path.isfile(os.path.join(image_dir, f)):
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(image_dir, f))
        self.annotation_dir = annotation_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load annotation
        annotation_path = os.path.join(self.annotation_dir, f"{image_name}.txt")
        boxes = []
        classes = []

        if os.path.isfile(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    l = int(parts[1])
                    t = int(parts[2])
                    w = int(parts[3])
                    h = int(parts[4])
                    boxes.append([l, t, l + w, t + h])
                    classes.append(cls_id)

        boxes = np.array(boxes)
        classes = np.array(classes)

        sample = {'image': image, 'boxes': boxes, 'classes': classes}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Transformation to resize images and adjust bounding boxes
class Resize:
    def __init__(self, size):
        self.size = size  # size should be (height, width)

    def __call__(self, sample):
        image, boxes, classes = sample['image'], sample['boxes'], sample['classes']
        w, h = image.size  # PIL Image size is (width, height)
        new_w, new_h = self.size

        # Resize image
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Scale bounding boxes
        if boxes.size > 0:
            boxes = boxes.astype(np.float32)
            scale_w = new_w / w
            scale_h = new_h / h
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h

        sample = {'image': image, 'boxes': boxes, 'classes': classes}
        return sample

# Transformation to convert PIL images and numpy arrays to tensors
class ToTensor:
    def __call__(self, sample):
        image, boxes, classes = sample['image'], sample['boxes'], sample['classes']
        image = T.ToTensor()(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        return {'image': image, 'boxes': boxes, 'classes': classes}

# Define the L2 loss function
def l2_loss(pred_boxes, gt_boxes):
    return F.mse_loss(pred_boxes, gt_boxes)

# Function to compute loss using IoU-based matching
def compute_loss(pred_boxes, gt_boxes):
    # Compute IoU matrix between predicted and ground truth boxes
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    if iou_matrix.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(pred_boxes.device)
    # Convert IoU to distances
    cost_matrix = 1 - iou_matrix.cpu().detach().numpy()

    # Solve the linear sum assignment problem (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pred_boxes = pred_boxes[row_ind]
    matched_gt_boxes = gt_boxes[col_ind]

    # Compute L2 loss between matched boxes
    loss = l2_loss(matched_pred_boxes, matched_gt_boxes)

    return loss

# Define class names (as per your original code)
class_names = {
    0: "People",
    1: "Bicycle",
    2: "Car",
    3: "Motorbike",
    5: "Bus",
    8: "Boat",
    15: "Cat",
    16: "Dog",
    39: "Bottle",
    41: "Cup",
    56: "Chair",
    60: "Table"
}

# Prepare data transformations and data loaders
transform = T.Compose([
    Resize((512, 512)),  # Resize images and adjust bounding boxes
    ToTensor()
])

# Custom collate function to handle variable-length tensors
def collate_fn(batch):
    images = []
    boxes = []
    classes = []
    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['boxes'])
        classes.append(sample['classes'])
    images = torch.stack(images, dim=0)
    return {'image': images, 'boxes': boxes, 'classes': classes}

train_dataset = ExDarkDataset(
    image_dir='DatasetExDark/ExDark_images/',
    annotation_dir='DatasetExDark/ExDark_Annno/',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,  # Adjust batch size as per GPU memory
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn  # Use the custom collate function
)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_model = UNet().to(device)
yolo_model = YOLO("yolo11n.pt").to(device)

# Utilize multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    unet_model = nn.DataParallel(unet_model)
    # Note: YOLO model may not support DataParallel directly due to its implementation
    # In practice, we might need to modify the YOLO model or handle batching differently

# Optimizer
optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    unet_model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        gt_boxes_batch = batch['boxes']  # List of tensors
        gt_classes_batch = batch['classes']  # List of tensors

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through U-Net
        processed_images = unet_model(images)

        # Clamp processed images to [0, 1]
        processed_images = torch.clamp(processed_images, 0, 1)

        # Convert processed_images to numpy and rearrange dimensions
        processed_images_np = processed_images.permute(0, 2, 3, 1).cpu().detach().numpy()

        # Run YOLO inference
        results = yolo_model(processed_images_np, imgsz=512, device=device, verbose=False)

        batch_loss = torch.tensor(0.0, requires_grad=True).to(device)

        for i, result in enumerate(results):
            pred_boxes = []

            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy().squeeze()
                pred_boxes.append(xyxy)

            pred_boxes = np.array(pred_boxes)
            pred_boxes = torch.as_tensor(pred_boxes, dtype=torch.float32).to(device)

            # Get ground truth boxes for this image
            gt_boxes_img = gt_boxes_batch[i].to(device)

            if pred_boxes.size(0) == 0 or gt_boxes_img.size(0) == 0:
                continue  # Skip if no boxes

            loss = compute_loss(pred_boxes, gt_boxes_img)
            batch_loss = batch_loss + loss

        if batch_loss == 0:
            continue  # Skip if no loss computed

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    elapsed = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Time: {elapsed:.2f}s")
