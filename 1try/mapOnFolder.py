import os

def process_annotations(base_folder1, base_folder2, process_fn):    
    # Get list of files in folder1
    files1 = sorted([f for f in os.listdir(base_folder1) if os.path.isfile(os.path.join(base_folder1, f))])

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
    return [list(map(float, line.split())) for line in data]

def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
    """Calculate precision and recall at a given IoU threshold."""
    tp, fp = 0, 0
    detected = set()
    
    for pred in pred_boxes:
        max_iou = 0
        max_iou_idx = -1
        for idx, gt in enumerate(true_boxes):
            if idx in detected:
                continue
            iou_value = iou(pred, gt)
            if iou_value > max_iou:
                max_iou = iou_value
                max_iou_idx = idx

        if max_iou >= iou_threshold:
            tp += 1
            detected.add(max_iou_idx)
        else:
            fp += 1

    fn = len(true_boxes) - len(detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

def calculate_map(folder1, folder2):
    all_precisions, all_recalls = [], []

    def process_fn(data1, data2):
        pred_boxes = parse_annotations(data1)
        true_boxes = parse_annotations(data2)

        print("Predictions:", pred_boxes)
        print("Ground Truth:", true_boxes)

        precision, recall = calculate_precision_recall(pred_boxes, true_boxes)
        all_precisions.append(precision)
        all_recalls.append(recall)

    process_annotations(folder1, folder2, process_fn)
    
    # Average precision and recall for mAP
    mAP = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    return mAP

# Example usage
mAP = calculate_map("results", "DatasetExDark/ExDark_Annno")
print("Mean Average Precision (mAP):", mAP)