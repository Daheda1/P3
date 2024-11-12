import os
from os import listdir
from os.path import isfile, join, splitext, basename
from ultralytics import YOLO
import cv2

# Define class ID to name mapping
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

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Define paths
input_path = "DatasetExDark/ExDark_images/"
save_path = "results/"

# Create the save directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Get all files in the input directory
onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
full_paths = [join(input_path, f) for f in onlyfiles]

# Run the model on the first 500 images with specified classes
results = model(
    full_paths[:50],
    classes=list(class_names.keys()),
    save=True,      # Saves the annotated images in the default 'runs' directory
    save_txt=True   # Saves the detection results as .txt files in the default 'runs' directory
)

# Iterate through each result
for result in results:
    image_path = result.path
    image_name = splitext(basename(image_path))[0]  # Extract filename without extension

    # Open the original image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        continue

    # Iterate through each detected box in the image
    for box in result.boxes:
        cls_id = int(box.cls.item())  # Class ID
        class_name = class_names.get(cls_id, "Unknown")  # Get class name
        xyxy = box.xyxy.squeeze().int().tolist()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = xyxy

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Prepare the label with class name
        label = class_name

        # Choose a font and get the size of the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw a filled rectangle behind the label for better visibility
        cv2.rectangle(
            image,
            (x_min, y_min - label_height - baseline),
            (x_min + label_width, y_min),
            (0, 255, 0),
            thickness=cv2.FILLED
        )

        # Put the class name text above the bounding box
        cv2.putText(
            image,
            label,
            (x_min, y_min - baseline),
            font,
            font_scale,
            (0, 0, 0),  # Text color (black)
            thickness,
            lineType=cv2.LINE_AA
        )

    # Save the annotated image to the save_path directory
    annotated_image_path = join(save_path, f"{image_name}_boxed.jpg")
    cv2.imwrite(annotated_image_path, image)
    print(f"Saved annotated image to {annotated_image_path}")

    # Additionally, save the detection results as a .txt file in save_path
    txt_file_path = join(save_path, f"{image_name}.txt")
    with open(txt_file_path, "w") as file:
        for box in result.boxes:
            cls_id = int(box.cls.item())  # Class ID
            xyxy = box.xyxy.squeeze().int().tolist()  # Bounding box coordinates
            custom_fields = [0] * 7  # Placeholder fields as per your format

            # **IMPORTANT FIX HERE**
            # Replace the hardcoded '1' with the actual class ID
            # If your format requires a specific format, adjust accordingly
            line = f"{cls_id} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]} {' '.join(map(str, custom_fields))}\n"
            file.write(line)

    print(f"Saved detection results to {txt_file_path}")
