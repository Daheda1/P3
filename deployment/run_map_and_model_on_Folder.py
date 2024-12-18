import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from modelparts.modelStructure import UNet
from modelparts.imagePreprocessing import scale_image, pad_image_to_target

def calculate_map_on_folder(image_folder, ground_truth_folder, yolo_model_path, unet_checkpoint_path, target_size=(640, 640)):
    """
    Calculates mAP for all images in a folder using a UNet model for preprocessing and a YOLO model for detection.
    
    Args:
        image_folder (str): Path to the folder containing input images.
        ground_truth_folder (str): Path to ground truth label files for evaluation.
        yolo_model_path (str): Path to the YOLO model.
        unet_checkpoint_path (str): Path to the UNet checkpoint.
        target_size (tuple): Target image size for preprocessing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load UNet model
    unet_model = UNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(unet_checkpoint_path, map_location=device)
    unet_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    unet_model.eval()

    # Load YOLO model
    yolo_model = YOLO(yolo_model_path).to(device)

    # Loop through images
    predictions = []
    with torch.no_grad():
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            if not image_name.endswith(('.jpg', '.png', '.jpeg')):
                continue

            # Load and preprocess image
            original_image = Image.open(image_path).convert("RGB")
            scaled_image, _ = scale_image(original_image, target_size, divideable_by=32)
            padded_image, _ = pad_image_to_target(scaled_image, target_size)
            input_tensor = torch.from_numpy(np.array(padded_image).astype(np.float32) / 255.0)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

            # UNet Inference
            enhanced_output = unet_model(input_tensor)
            enhanced_image = enhanced_output.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)

            # YOLO Detection
            yolo_results = yolo_model(enhanced_image)
            predictions.append(yolo_results)

    # Evaluate mAP
    metrics = yolo_model.val(data={"val": ground_truth_folder}, save_json=False)
    print("Mean Average Precision (mAP):", metrics.box.map)

# Example usage:
if __name__ == "__main__":
    image_folder = "path_to_your_images"
    ground_truth_folder = "path_to_ground_truth_labels"
    yolo_model_path = "yolo11n.pt"
    unet_checkpoint_path = "checkpoint_epoch_7.pth"

    calculate_map_on_folder(image_folder, ground_truth_folder, yolo_model_path, unet_checkpoint_path)