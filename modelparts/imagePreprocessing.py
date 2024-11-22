
from PIL import Image, ImageOps
import logging

from PIL import Image

def scale_image(image, target_size, divideable_by=32):
    """
    Scales an image so that one dimension matches the target size and maintains aspect ratio.
    Ensures scaled dimensions are divisible by a specified value.

    Args:
        image (PIL.Image.Image): The input image to be scaled.
        target_size (tuple): The target size (width, height) for scaling.
        divideable_by (int): Value by which dimensions must be divisible.

    Returns:
        PIL.Image.Image: The scaled image.
        tuple: The scaling ratio used for resizing.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate scaling factors for each dimension
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # Choose the smaller scaling factor to fit one dimension
    resize_ratio = min(width_ratio, height_ratio)

    # Scale dimensions while ensuring divisibility by `divideable_by`
    new_width = int((original_width * resize_ratio) // divideable_by) * divideable_by
    new_height = int((original_height * resize_ratio) // divideable_by) * divideable_by

    #print(f"Original size: ({original_width}, {original_height})")
    #print(f"Scaled size: ({new_width}, {new_height})")

    # Resize the image
    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return scaled_image, resize_ratio


def scale_bounding_boxes(ground_truth, resize_ratio, padding, divideable_by=32):
    """
    Scales bounding boxes to match the image scaling factor and ensures dimensions are divisible by a specified value.

    Args:
        ground_truth (list): List of bounding boxes, where each box is [label, l, t, w, h].
        resize_ratio (float): Scaling factor applied to the bounding boxes.
        divideable_by (int): Value by which bounding box dimensions must be divisible.

    Returns:
        list: Scaled bounding boxes with dimensions adjusted to be divisible by `divideable_by`.
    """
    adjusted_ground_truth = []

    for bbox in ground_truth:
        if not isinstance(bbox, list) or len(bbox) != 5:
            logging.error(f"Invalid bounding box format: {bbox}")
            continue  # Skip invalid boxes

        label, l, t, w, h = bbox

        # Scale coordinates and dimensions
        l_scaled = l * resize_ratio
        t_scaled = t * resize_ratio
        w_scaled = w * resize_ratio
        h_scaled = h * resize_ratio

        # Shift coordinates based on padding
        l_padded = l_scaled + padding[0]
        t_padded = t_scaled + padding[1]

        # Adjust dimensions to be divisible by `divideable_by`
        w_scaled = round(w_scaled / divideable_by) * divideable_by
        h_scaled = round(h_scaled / divideable_by) * divideable_by

        # Convert to integers
        adjusted_bbox = [
            label,
            int(round(l_padded)),
            int(round(t_padded)),
            int(round(w_scaled)),
            int(round(h_scaled))
        ]
        adjusted_ground_truth.append(adjusted_bbox)

    return adjusted_ground_truth


def pad_image_to_target(image, target_size):
    """
    Pads an image to center it within the target size.

    Args:
        image (PIL.Image.Image): The input image to be padded.
        target_size (tuple): The target size (width, height) for the output image.

    Returns:
        PIL.Image.Image: The padded image.
        tuple: Padding values (pad_left, pad_top, pad_right, pad_bottom).
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate padding
    pad_left = (target_width - original_width) // 2
    pad_top = (target_height - original_height) // 2
    pad_right = target_width - original_width - pad_left
    pad_bottom = target_height - original_height - pad_top

    # Apply padding
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    padded_image = ImageOps.expand(image, border=padding, fill=(0, 0, 0))  # Fill with black (or specify a color)

    return padded_image, padding


from PIL import Image, ImageDraw

def draw_bounding_boxes(image, bounding_boxes):
    """
    Draws bounding boxes on an image and displays it.

    Args:
        image (PIL.Image.Image): The image to draw bounding boxes on.
        bounding_boxes (list): A list of bounding boxes, where each box is [label, l, t, w, h].

    Returns:
        PIL.Image.Image: The image with bounding boxes drawn.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw each bounding box
    for bbox in bounding_boxes:
        if len(bbox) != 5:
            print(f"Skipping invalid bounding box: {bbox}")
            continue

        label, l, t, w, h = bbox

        # Define the rectangle coordinates
        left = l
        top = t
        right = l + w
        bottom = t + h

        # Draw the rectangle
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # Optionally, draw the label
        draw.text((left, top), str(label), fill="yellow")

    # Show the image
    image.show()

    return image

if __name__ == "__main__":

    # Example usage
    image = Image.open("DatasetExDark/ExDark_images/2015_00002.png")  # Load an image
    target_size = (1280, 1280)  # Target size
    scaled_image, resize_ratio = scale_image(image, target_size)

    scaled_image.show()  # Show the resized image

    padded_image, padding = pad_image_to_target(scaled_image, target_size)
    # Output the padding values
    print("Padding:", padding)

    # Show the padded image
    padded_image.show()


    ground_truth = [
        [1, 136, 190, 79, 109],  # Label 1, x=136, y=190, width=79, height=109
        [1, 219, 172, 63, 131],  # Label 1, x=219, y=172, width=63, height=131
        [1, 277, 188, 76, 124],  # Label 1, x=277, y=188, width=76, height=124
        [1, 348, 183, 57, 81],   # Label 1, x=348, y=183, width=57, height=81
        [2, 316, 171, 33, 26],   # Label 2, x=316, y=171, width=33, height=26
    ]

    # Scale bounding boxes
    adjusted_ground_truth = scale_bounding_boxes(ground_truth, resize_ratio, padding)

    # Output adjusted bounding boxes
    print(adjusted_ground_truth)

    draw_bounding_boxes(padded_image, adjusted_ground_truth)