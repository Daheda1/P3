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

# Function to draw dashed rectangles
def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
    """
    Draws a dashed rectangle between pt1 and pt2 on the image.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Top edge
    draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
    # Bottom edge
    draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_length, gap_length)
    # Left edge
    draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_length, gap_length)
    # Right edge
    draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
    """
    Draws a dashed line between pt1 and pt2 on the image.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Calculate the total length of the line
    total_length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    # Calculate the number of dashes
    dash_gap = dash_length + gap_length
    num_dashes = int(total_length // dash_gap)

    for i in range(num_dashes + 1):
        start_ratio = (dash_gap * i) / total_length
        end_ratio = (dash_gap * i + dash_length) / total_length
        if end_ratio > 1:
            end_ratio = 1
        start_x = int(x1 + (x2 - x1) * start_ratio)
        start_y = int(y1 + (y2 - y1) * start_ratio)
        end_x = int(x1 + (x2 - x1) * end_ratio)
        end_y = int(y1 + (y2 - y1) * end_ratio)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

# Load the YOLO model
model = YOLO("yolo11n.pt")  # Ensure this path is correct

# Define paths
input_path = "DatasetExDark/ExDark_images/"
annotation_path = "DatasetExDark/ExDark_Annno/"
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
    save=False,      # We'll handle saving manually
    save_txt=False   # We'll handle saving txt manually
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

    # Draw detection bounding boxes with labels on top
    for box in result.boxes:
        cls_id = int(box.cls.item())  # Class ID
        class_name = class_names.get(cls_id, "Unknown")  # Get class name
        xyxy = box.xyxy.squeeze().int().tolist()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = xyxy

        # Draw the detection bounding box (Solid Green)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Prepare the label with class name
        label = class_name

        # Choose a font and get the size of the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_text = 1
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness_text)

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
            thickness_text,
            lineType=cv2.LINE_AA
        )

    # Load ground truth annotations
    gt_annotation_file = join(annotation_path, f"{image_name}.txt")
    if os.path.isfile(gt_annotation_file):
        with open(gt_annotation_file, "r") as gt_file:
            for line in gt_file:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # Skip invalid lines
                try:
                    gt_cls_id = int(parts[0])
                    gt_l = int(parts[1])
                    gt_t = int(parts[2])
                    gt_w = int(parts[3])
                    gt_h = int(parts[4])

                    gt_class_name = class_names.get(gt_cls_id, "Unknown")

                    # Calculate x_max and y_max from l, t, w, h
                    gt_x_min = gt_l
                    gt_y_min = gt_t
                    gt_x_max = gt_l + gt_w
                    gt_y_max = gt_t + gt_h

                    # Draw the ground truth bounding box (Striped Red)
                    draw_dashed_rectangle(
                        image,
                        (gt_x_min, gt_y_min),
                        (gt_x_max, gt_y_max),
                        color=(0, 0, 255),  # Red color
                        thickness=2,
                        dash_length=10,
                        gap_length=5
                    )

                    # Prepare the label with class name
                    gt_label = gt_class_name

                    # Choose a font and get the size of the label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness_text = 1
                    (gt_label_width, gt_label_height), gt_baseline = cv2.getTextSize(gt_label, font, font_scale, thickness_text)

                    # Calculate position for label at the bottom of the bounding box
                    label_x = gt_x_min
                    label_y = gt_y_max + gt_label_height + gt_baseline

                    # Ensure the label does not go outside the image
                    label_y = min(label_y, image.shape[0] - gt_label_height - gt_baseline)

                    # Draw a filled rectangle behind the label for better visibility
                    cv2.rectangle(
                        image,
                        (label_x, label_y - gt_label_height - gt_baseline),
                        (label_x + gt_label_width, label_y),
                        (0, 0, 255),  # Red color
                        thickness=cv2.FILLED
                    )

                    # Put the class name text below the bounding box
                    cv2.putText(
                        image,
                        gt_label,
                        (label_x, label_y - gt_baseline),
                        font,
                        font_scale,
                        (255, 255, 255),  # Text color (white)
                        thickness_text,
                        lineType=cv2.LINE_AA
                    )
                except ValueError:
                    print(f"Invalid annotation in {gt_annotation_file}: {line}")
    else:
        print(f"Ground truth annotation file not found for {image_name}")

    # Save the annotated image to the save_path directory
    annotated_image_path = join(save_path, f"{image_name}_boxed.jpg")
    cv2.imwrite(annotated_image_path, image)
    print(f"Saved annotated image to {annotated_image_path}")

    # Additionally, save the detection results as a .txt file in save_path
    txt_file_path = join(save_path, f"{image_name}.txt")
    with open(txt_file_path, "w") as file:
        for box in result.boxes:
            cls_id = int(box.cls.item())  # Class ID
            xyxy = box.xyxy.squeeze().int().tolist()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
            
            # Convert detection format to match ground truth
            x_min, y_min, x_max, y_max = xyxy
            width = x_max - x_min
            height = y_max - y_min
            
            # Format the line to match ground truth: <class_id> <l> <t> <w> <h> 0 0 0 0 0 0 0
            line = f"{cls_id} {x_min} {y_min} {width} {height} 0 0 0 0 0 0 0\n"
            file.write(line)

    print(f"Saved detection results to {txt_file_path}")

