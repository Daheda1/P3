import os

def process_annotations(base_folder1, base_folder2, process_fn):
    """
    Processes annotation files from two directories, applying a processing function
    to each pair of corresponding files. Only `.txt` files from base_folder1 are loaded.

    Args:
        base_folder1 (str): Path to the first directory (e.g., predicted annotations).
        base_folder2 (str): Path to the second directory (e.g., ground truth annotations).
        process_fn (callable): Function to process each pair of annotation data.
    """
    # Get list of .txt files in folder1
    files1 = sorted([
        f for f in os.listdir(base_folder1)
        if os.path.isfile(os.path.join(base_folder1, f)) and f.lower().endswith('.txt')
    ])

    if not files1:
        print(f"No '.txt' files found in {base_folder1}.")
        return

    # Run the process function on each file that exists in both folders
    for filename in files1:
        path1 = os.path.join(base_folder1, filename)
        path2 = os.path.join(base_folder2, filename)

        # Check if the file exists in the second folder
        if not os.path.exists(path2):
            print(f"File {filename} not found in {base_folder2}. Skipping.")
            continue

        # Open and read data from both files
        with open(path1, 'r') as file1, open(path2, 'r') as file2:
            data1 = [line.strip() for line in file1.readlines()]
            data2 = [line.strip() for line in file2.readlines()]

            # Apply process_fn to corresponding files' data
            process_fn(data1, data2)

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    # Convert center, width, height to x_min, y_min, x_max, y_max
    x1_min, y1_min = box1[1] - box1[3] / 2, box1[2] - box1[4] / 2
    x1_max, y1_max = box1[1] + box1[3] / 2, box1[2] + box1[4] / 2
    x2_min, y2_min = box2[1] - box2[3] / 2, box2[2] - box2[4] / 2
    x2_max, y2_max = box2[1] + box2[3] / 2, box2[2] + box2[4] / 2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def parse_annotations(data):
    """Parse annotation lines into structured lists of bounding boxes."""
    return [list(map(float, line.split())) for line in data if line.strip()]

def calculate_mae(folder1, folder2, iou_threshold=0.5):
    """
    Calculates the Mean Absolute Error (MAE) for bounding box parameters between
    predicted and ground truth annotations.

    Args:
        folder1 (str): Directory containing predicted annotation `.txt` files.
        folder2 (str): Directory containing ground truth annotation `.txt` files.
        iou_threshold (float): IoU threshold to consider a prediction as a match.

    Returns:
        list: MAE for [x_center, y_center, width, height], or None if no matches found.
    """
    total_errors = [0.0, 0.0, 0.0, 0.0]  # For x_center, y_center, width, height
    count = 0

    def process_fn(data1, data2):
        nonlocal total_errors, count
        pred_boxes = parse_annotations(data1)
        true_boxes = parse_annotations(data2)

        print(f"Processing File: Comparing {len(pred_boxes)} Predictions with {len(true_boxes)} Ground Truths")

        matched_gt = set()
        matched_pred = set()

        # Match predictions to ground truth using IoU
        for pred_idx, pred in enumerate(pred_boxes):
            max_iou = 0
            max_iou_idx = -1
            for gt_idx, gt in enumerate(true_boxes):
                if gt_idx in matched_gt:
                    continue
                current_iou = iou(pred, gt)
                if current_iou > max_iou:
                    max_iou = current_iou
                    max_iou_idx = gt_idx
            if max_iou >= iou_threshold and max_iou_idx != -1:
                matched_gt.add(max_iou_idx)
                matched_pred.add(pred_idx)
                # Compute absolute differences for each parameter
                for i in range(1, 5):  # x_center, y_center, width, height
                    error = abs(pred[i] - true_boxes[max_iou_idx][i])
                    total_errors[i-1] += error
                count += 1
            else:
                # If no match, consider the entire box as error
                for i in range(1, 5):
                    # Assuming the maximum possible error is the image size; adjust as needed
                    max_error = 1.0  # Placeholder value; adjust based on your data scale
                    total_errors[i-1] += max_error
                count += 1

        # Handle unmatched ground truth boxes
        unmatched_gt = set(range(len(true_boxes))) - matched_gt
        for gt_idx in unmatched_gt:
            for i in range(1, 5):
                max_error = 1.0  # Placeholder value; adjust based on your data scale
                total_errors[i-1] += max_error
            count += 1

    process_annotations(folder1, folder2, process_fn)

    if count == 0:
        print("No matched boxes found. MAE is undefined.")
        return None

    mae = [error / count for error in total_errors]
    return mae

# Example usage
if __name__ == "__main__":
    folder1 = "results"  # Predicted annotations (only .txt files will be loaded)
    folder2 = "DatasetExDark/ExDark_Annno"  # Ground truth annotations
    mae = calculate_mae(folder1, folder2)
    if mae is not None:
        print("Mean Absolute Error (MAE) for [x_center, y_center, width, height]:", mae)
