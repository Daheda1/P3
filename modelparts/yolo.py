# modelparts/yolo.py

from ultralytics import YOLO
import logging

# Initialize the YOLO model globally to reuse across function calls
yolo_model = YOLO("yolo11n.pt")

def yolo_object(tensor_image, class_ids):
    """
    Performs YOLO object detection on a Torch Tensor image.

    Args:
        tensor_image (torch.Tensor): Image tensor in BCHW format with RGB channels, normalized [0.0, 1.0].
        class_ids (list): List of class IDs to filter detections.

    Returns:
        dict: Cumulative confidence scores for each specified class ID.
    """
    if tensor_image is None or tensor_image.numel() == 0:
        logging.error("Invalid tensor image input for YOLO object detection.")
        return {}
    
    # Ensure the tensor is on CPU and detached from the computation graph
    tensor_image_cpu = tensor_image.cpu()
    
    # YOLO expects images in BCHW format; ensure the batch size is maintained
    if tensor_image_cpu.dim() == 3:
        tensor_image_cpu = tensor_image_cpu.unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    results = yolo_model(tensor_image_cpu, conf=0.1, classes=class_ids)
    
    # Initialize a dictionary to store cumulative confidence scores
    confidence_scores = {cls_id: 0.0 for cls_id in class_ids}
    
    # Iterate over detections
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_id in class_ids:
                    confidence_scores[cls_id] += conf
                    logging.debug(f"Detected class {cls_id} with confidence {conf}.")
    
    return confidence_scores