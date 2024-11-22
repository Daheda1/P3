import torch

def crop_bounding_boxes(image: torch.Tensor, boxes: list, overlap_threshold: float = 0.5):
    """
    Crops image regions based on provided bounding boxes and identifies overlapping objects via IoU.
    The overlap is determined using the intersection area between two boxes. If the intersection area exceeds
    50% of the second box’s area, the corresponding object’s label is considered overlapping.

    Args:
        image (torch.Tensor): Image tensor in B x C x H x W format.
        boxes (list): List of lists of bounding boxes for each image in the batch.
                      Each element is a list of bounding boxes for the corresponding image,
                      where each bounding box is in the format [label, l, t, w, h].
        overlap_threshold (float): Threshold for determining overlap. Default is 0.5.

    Returns:
        list: List of torch.Tensor: Cropped image tensors in BCHW format with batch size always 1.
        list: List of lists containing labels of overlapping objects for each crop.
    """
    crops = []
    overlapping_labels_list = []

    image = image / 255.0

    B, C, H, W = image.shape

    for b in range(B):
        image_b = image[b]  # Shape: [C, H, W]
        boxes_b = boxes[b]  # List of boxes for image b

        show_tensor_image(image_b)

        labels = [box[0] for box in boxes_b]
        box_coords = [box[1:] for box in boxes_b]  # [l, t, w, h]
        num_boxes = len(boxes_b)

        for i in range(num_boxes):
            label1 = labels[i]
            l1, t1, w1, h1 = box_coords[i]
            r1 = l1 + w1
            b1 = t1 + h1

            # Clamp the coordinates to be within image boundaries
            l1_clamped = max(0, l1)
            t1_clamped = max(0, t1)
            r1_clamped = min(W, r1)
            b1_clamped = min(H, b1)

            # Crop the image
            crop = image_b[:, t1_clamped:b1_clamped, l1_clamped:r1_clamped]

            #show_tensor_image(crop)

            # Handle empty crops
            if crop.shape[1] == 0 or crop.shape[2] == 0:
                crop = torch.zeros((C, 1, 1), dtype=image.dtype, device=image.device)

            # Add batch dimension back
            crop = crop.unsqueeze(0)  # Shape: [1, C, H_i, W_i]
            crops.append(crop)

            # Initialize overlapping labels list
            overlapping_labels = []

            for j in range(num_boxes):
                if i == j:
                    continue

                label2 = labels[j]
                l2, t2, w2, h2 = box_coords[j]
                r2 = l2 + w2
                b2 = t2 + h2

                # Compute intersection coordinates
                inter_left = max(l1, l2)
                inter_top = max(t1, t2)
                inter_right = min(r1, r2)
                inter_bottom = min(b1, b2)

                # Compute intersection area
                inter_width = inter_right - inter_left
                inter_height = inter_bottom - inter_top

                if inter_width > 0 and inter_height > 0:
                    inter_area = inter_width * inter_height

                    # Compute area of box2 (the second box)
                    area2 = w2 * h2

                    # Compute overlap ratio
                    overlap_ratio = inter_area / area2

                    if overlap_ratio > overlap_threshold:
                        overlapping_labels.append(label2)

            # Add overlapping labels to the list
            overlapping_labels_list.append(overlapping_labels)

    return crops, overlapping_labels_list



import torch
import matplotlib.pyplot as plt
import numpy as np

def show_tensor_image(tensor):
    """
    Displays a torch.Tensor as an image using Matplotlib.

    Args:
        tensor (torch.Tensor): The image tensor to display.
                               Expected shape is [C, H, W] or [B, C, H, W].
    """
    # Ensure the tensor is on CPU and detach from computation graph
    tensor = tensor.detach().cpu()

    # Handle batch dimension
    if tensor.dim() == 4:
        # Select the first image in the batch
        tensor = tensor[0]

    # Convert tensor to NumPy array
    image = tensor.numpy()

    # Check for grayscale image
    if image.shape[0] == 1:
        # Remove the channel dimension
        image = image.squeeze(0)
        plt.imshow(image, cmap='gray')
    else:
        # Transpose from [C, H, W] to [H, W, C]
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)

    plt.axis('off')
    plt.show()