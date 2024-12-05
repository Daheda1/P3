# modelparts/yolo.py

from ultralytics import YOLO
import torch
import logging
from PIL import Image
import torchvision.transforms.functional as TF

# Initialize the YOLO model globally to reuse across function calls
yolo_model = YOLO("yolo11n.pt")

def yolo_object(tensor_image, class_ids=[1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]):
    """
    Performs YOLO object detection on a Torch Tensor image.

    Args:
        tensor_image (torch.Tensor): Image tensor in BCHW format with RGB channels, normalized [0.0, 1.0].
        class_ids (list): List of class IDs to filter detections.

    Returns:
        list of list of dict: A nested list where each inner list corresponds to detections for an image.
                              Each detection is a dictionary with 'class_id', 'confidence', and 'bbox'.
    """
    tensor_image = tensor_image


    if tensor_image is None or tensor_image.numel() == 0:
        logging.error("Invalid tensor image input for YOLO object detection.")
        return []

    # Ensure the tensor is on CPU and detached from the computation graph
    tensor_image_cpu = tensor_image.cpu().detach()

    # YOLO expects images in BCHW format; ensure the batch size is maintained
    if tensor_image_cpu.dim() == 3:
        tensor_image_cpu = tensor_image_cpu.unsqueeze(0)  # Add batch dimension

    #visualize_image(tensor_image)
    #visualize_image(tensor_image_cpu)

    # Perform inference
    results = yolo_model(tensor_image_cpu, conf=0.5, classes=class_ids, verbose=True)
    
    all_detections = []

    for result in results:
        image_detections = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                # Extract attributes safely
                try:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                except AttributeError as e:
                    logging.error(f"Error parsing detection box: {e}")
                    continue

                # Filter detections by class ID
                if cls_id in class_ids:
                    image_detections.append({
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    logging.debug(f"Detected class {cls_id} with confidence {conf}, bbox: {[x1, y1, x2, y2]}")
        else:
            logging.info("No detections found for this image.")

        all_detections.append(image_detections)

    return all_detections

import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_image(image_tensor, output_dir="visualizations", batch_idx=0, save_only=False):
    """
    Visualizes or saves an image tensor. Converts the tensor to numpy format and handles normalization.

    Parameters:
        image_tensor (torch.Tensor): The image tensor to visualize or save. Shape should be [C, H, W].
        output_dir (str): Directory to save the images if save_only is True.
        batch_idx (int): Index of the batch, used for naming saved images.
        save_only (bool): If True, only saves the image; otherwise, displays it.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check tensor dimensions
    if image_tensor.dim() == 4:
        # If it's a batch, take the first image
        image_tensor = image_tensor[0]
    elif image_tensor.dim() != 3:
        raise ValueError(f"Expected image_tensor with 3 or 4 dimensions, but got {image_tensor.dim()}.")

    # Convert tensor to numpy format
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]

    # Normalize image values if they are not in range [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    # Save or display the image
    file_path = os.path.join(output_dir, f"image_batch_{batch_idx}.png")
    if save_only:
        plt.imsave(file_path, image_np)
    else:
        plt.imshow(image_np)
        plt.axis("off")
        plt.title(f"Batch {batch_idx}")
        plt.show()

    print(f"Image saved to: {file_path}" if save_only else "Image displayed.")


if __name__ == "__main__":
    # Prepare your input tensor
    image_path = "DatasetExDark/ExDark_images/2015_07353.png"
    pil_image = Image.open(image_path)
    input_tensor = TF.to_tensor(pil_image).unsqueeze(0)  # Add batch dimension
    input_tensor = TF.resize(input_tensor, [640, 640])  # Resize to YOLO's input size

    # Enable gradient computation for the input tensor
    detections = yolo_object(input_tensor, class_ids=[1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60])
    print(detections)