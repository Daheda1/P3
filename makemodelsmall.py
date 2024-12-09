import torch
from torch import nn

# Load the model data
model_data = torch.load("yolo11n.pt")
model = model_data["model"]  # Extract the model

# Locate the `Detect` layer
detect_layer = model.model[-1]  # Access the Detect layer

# Update the number of classes
num_classes = 10  # Example: New number of classes
num_bounding_boxes = detect_layer.cv2[0][2].out_channels // (5 + detect_layer.nc)  # Calculate number of bounding boxes

# Update attributes of the `Detect` layer
detect_layer.nc = num_classes  # Update the number of classes
detect_layer.no = (num_classes + 5) * num_bounding_boxes  # Update the output channels

# Modify each scale in `cv2`
for i, sequential in enumerate(detect_layer.cv2):
    # Modify the last Conv2d layer in each sequential block
    in_channels = sequential[2].in_channels  # Input channels of the last Conv2d
    detect_layer.cv2[i][2] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=detect_layer.no,
        kernel_size=1,
        stride=1,
        padding=0
    )

# Modify each scale in `cv3`
for i, sequential in enumerate(detect_layer.cv3):
    # Modify the last Conv2d layer in each sequential block
    in_channels = sequential[1][1].in_channels  # Input channels of the last Conv2d in `cv3`
    detect_layer.cv3[i][2] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=detect_layer.no,
        kernel_size=1,
        stride=1,
        padding=0
    )

# Save the modified model
model_data["model"] = model
torch.save(model_data, "yolo11n_modified.pt")