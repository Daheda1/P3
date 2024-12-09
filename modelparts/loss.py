from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
import logging


def _compute_single_loss(image, ground_truth, yolo_model):
    """
    Computes YOLO loss for a single set of images.

    Args:
        image (torch.Tensor): Batch of input images in BCHW format with RGB channels, normalized [0.0, 1.0].
        ground_truth (list): List of bounding boxes and classes in format [class, l, t, w, h].
        model (YOLO): The YOLO model instance.

    Returns:
        torch.Tensor: The computed loss with grad_fn.
    """
    # Ensure the image tensor is in the correct format and requires grad
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension if missing
    image = image.requires_grad_()
    image = image.to(yolo_model.device)

    # Prepare lists for class labels and bounding boxes
    cls_list = []
    bboxes_list = []
    batch_idx_list = []

    for i, gt in enumerate(ground_truth):
        for gt_item in gt:
            class_id, x_center, y_center, width, height = gt_item

            cls_list.append([class_id])  # Class IDs as [class_id]
            bboxes_list.append([x_center, y_center, width, height])
            batch_idx_list.append(i)  # Batch index

    # Convert lists to tensors
    cls_tensor = torch.tensor(cls_list, dtype=torch.float32, device=image.device)
    bboxes_tensor = torch.tensor(bboxes_list, dtype=torch.float32, device=image.device)
    batch_idx_tensor = torch.tensor(batch_idx_list, dtype=torch.int64, device=image.device)

    # Prepare the target batch dictionary
    train_batch = {
        'cls': cls_tensor,            # Shape: [num_boxes, 1]
        'bboxes': bboxes_tensor,      # Shape: [num_boxes, 4]
        'batch_idx': batch_idx_tensor # Shape: [num_boxes]
    }

    # Perform forward pass
    pred = yolo_model.model(image)

    # Compute loss using the custom loss function
    loss = yolo_model.model.loss(pred, train_batch)[0]  # Access the total loss

    return loss



def calculate_loss(enhanced_image, ground_truth, yolo_model):
    """
    Calculates the difference in YOLO loss between enhanced and original images.
s
    Args:
        enhanced_image (torch.Tensor): Batch of enhanced images in BCHW format.
        original_image (torch.Tensor): Batch of original images in BCHW format.
        ground_truth (list): List of bounding boxes and classes.
        class_ids (list): List of class IDs.

    Returns:
        torch.Tensor: The difference in loss (enhanced - original).
    """
    # Compute loss for enhanced images
    loss_enhanced = _compute_single_loss(enhanced_image, ground_truth, yolo_model)

    return loss_enhanced

def load_image(image_path, target_size=(640, 640)):
    """
    Load an image from the specified path and preprocess it.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size as (height, width).

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [C, H, W].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(target_size),            # Resize image
        transforms.ToTensor(),                    # Convert to tensor and scale [0, 1]
        # Optionally, add normalization
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

    # Open and transform the image
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    image_tensor = transform(image)                # Shape: [3, H, W]

    # Ensure the tensor requires gradients if needed
    image_tensor.requires_grad = True

    return image_tensor