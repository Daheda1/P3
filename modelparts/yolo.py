from ultralytics import YOLO
import torch
import logging
from PIL import Image
import torchvision.transforms.functional as TF

# Initialize the YOLO model globally to reuse across function calls
yolo_model = YOLO("yolo11n.pt")

#yolo_model.model.requires_grad_()
#yolo_model.model.train() 

def yolo_object(tensor_image, class_ids):
    """
    Performs YOLO object detection on a Torch Tensor image.

    Args:
        tensor_image (torch.Tensor): Image tensor in BCHW format with RGB channels, normalized [0.0, 1.0].
        class_ids (list): List of class IDs to filter detections.

    Returns:
        torch.Tensor: Cumulative confidence scores for each specified class ID.
    """
    if tensor_image is None or tensor_image.numel() == 0:
        logging.error("Invalid tensor image input for YOLO object detection.")
        return torch.zeros(len(class_ids), dtype=torch.float32, device=tensor_image.device)
    
    # Ensure the tensor is in BCHW format; add batch dimension if needed
    if tensor_image.dim() == 3:
        tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension

    # Perform inference
    results = yolo_model(tensor_image, conf=0.1, classes=class_ids)

    # Initialize cumulative confidence scores as a tensor
    confidence_scores = torch.zeros(len(class_ids), dtype=torch.float32, device=tensor_image.device)
    
    # Iterate over detections
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0]
                if cls_id in class_ids:
                    index = class_ids.index(cls_id)
                    confidence_scores[index] += conf
    
    return confidence_scores


if __name__ == "__main__":

    # Prepare your input tensor
    image_path = "DatasetExDark/ExDark_images/2015_07353.png"
    pil_image = Image.open(image_path)
    input_tensor = TF.to_tensor(pil_image).unsqueeze(0)  # Add batch dimension
    input_tensor = TF.resize(input_tensor, [640, 640])  # Resize to YOLO's input size

    # Enable gradient computation for the input tensor
    input_tensor.requires_grad_()


def convert_to_yolo_format(bboxes, image_width, image_height):
    """
    Convert bounding box list to YOLO format.
    
    Args:
        bboxes (list): List of bounding boxes in format [class, l, t, w, h].
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    
    Returns:
        list: Bounding boxes in YOLO format [class, x_center, y_center, width, height].
    """
    yolo_bboxes = []
    for bbox in bboxes:
        class_id, l, t, w, h = bbox
        x_center = (l + w / 2) / image_width
        y_center = (t + h / 2) / image_height
        norm_width = w / image_width
        norm_height = h / image_height
        yolo_bboxes.append([class_id, x_center, y_center, norm_width, norm_height])
    return yolo_bboxes

# Example input
bboxes = [[1, 322, 470, 192, 352], [2, 566, 400, 320, 320]]
image_width = 1024
image_height = 768

# Convert
yolo_bboxes = convert_to_yolo_format(bboxes, image_width, image_height)
print(yolo_bboxes)