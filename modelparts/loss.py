from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# Load YOLO model and set it to training mode
model = YOLO("yolo11n.pt")
model.model.requires_grad_()
model.model.train()
model.model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

def _compute_single_loss(image, ground_truth, class_ids, model):
    """
    Computes YOLO loss for a single set of images.

    Args:
        image (torch.Tensor): Batch of input images in BCHW format with RGB channels, normalized [0.0, 1.0].
        ground_truth (list): List of bounding boxes and classes in format [class, l, t, w, h].
        class_ids (list): List of class IDs.
        model (YOLO): The YOLO model instance.

    Returns:
        torch.Tensor: The computed loss with grad_fn.
    """
    # Ensure the image tensor is in the correct format and requires grad
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension if missing
    image = image.requires_grad_()

    # Get image dimensions
    _, _, img_height, img_width = image.shape

    # Prepare lists for class labels and bounding boxes
    cls_list = []
    bboxes_list = []
    batch_idx_list = []

    for gt in ground_truth:
        for gt_item in gt:
            class_id, l, t, w, h = gt_item

            # Normalize bounding box coordinates to [0, 1]
            x_center = (l + w / 2) / img_width
            y_center = (t + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height

            cls_list.append([class_id])  # Class IDs as [class_id]
            bboxes_list.append([x_center, y_center, norm_width, norm_height])
            batch_idx_list.append(0)  # Assuming single image per batch item

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
    pred = model.model(image)

    # Compute loss using YOLO's internal loss function
    loss = model.model.loss(train_batch, pred)[0]  # Access the total loss

    return loss

def calculate_loss(enhanced_image, original_image, ground_truth, class_ids=[1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]):
    """
    Calculates the difference in YOLO loss between enhanced and original images.

    Args:
        enhanced_image (torch.Tensor): Batch of enhanced images in BCHW format.
        original_image (torch.Tensor): Batch of original images in BCHW format.
        ground_truth (list): List of bounding boxes and classes.
        class_ids (list): List of class IDs.

    Returns:
        torch.Tensor: The difference in loss (enhanced - original).
    """
    # Compute loss for enhanced images
    loss_enhanced = _compute_single_loss(enhanced_image, ground_truth, class_ids, model)

    # Compute loss for original images
    loss_original = _compute_single_loss(original_image, ground_truth, class_ids, model)

    # Calculate the difference
    loss_diff = loss_enhanced - loss_original

    return loss_diff

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

if __name__ == "__main__":
    # Define the paths to your images
    original_image_path = "DatasetExDark/ExDark_images/2015_00012.jpg"  # <-- Replace with your original image path
    enhanced_image_path = "DatasetExDark/ExDark_images/2015_00003.png"  # <-- Replace with your enhanced image path

    try:
        # Load and preprocess the original image
        original_image = load_image(original_image_path, target_size=(640, 640))
        # Load and preprocess the enhanced image
        enhanced_image = load_image(enhanced_image_path, target_size=(640, 640))
    except Exception as e:
        print(f"Error loading images: {e}")
        exit(1)

    # Stack images into batches if you have multiple images
    # For single image pair, add batch dimension
    original_image_batch = original_image.unsqueeze(0)  # Shape: [1, 3, 640, 640]
    enhanced_image_batch = enhanced_image.unsqueeze(0)  # Shape: [1, 3, 640, 640]

    # Example ground truth data
    ground_truth = [
        [
            [1, 322, 470, 192, 352],
            [2, 566, 400, 320, 320]
        ]
    ]

    # Call the loss calculation function
    loss_difference = calculate_loss(enhanced_image_batch, original_image_batch, ground_truth)

    # Interpret the result
    if loss_difference.item() < 0:
        print(f"Enhanced image has lower loss by {abs(loss_difference.item())}")
    elif loss_difference.item() > 0:
        print(f"Enhanced image has higher loss by {loss_difference.item()}")
    else:
        print("Enhanced and original images have the same loss.")

    # Additionally, print the loss difference
    print("Loss Difference (Enhanced - Original):", loss_difference.item())