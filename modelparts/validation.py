import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from modelparts.loss import calculate_loss
from modelparts.yolo import yolo_object
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
from modelparts.loadData import ExDarkDataset, custom_collate_fn, ExDark

def box_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def match_predictions_to_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    tp_per_class = {}
    fp_per_class = {}
    fn_per_class = {}

    gt_by_class = {}
    for b, l in zip(gt_boxes, gt_labels):
        gt_by_class.setdefault(l.item(), []).append(b)

    pred_by_class = {}
    for b, l in zip(pred_boxes, pred_labels):
        pred_by_class.setdefault(l.item(), []).append(b)

    all_classes = set(gt_by_class.keys()).union(set(pred_by_class.keys()))
    for c in all_classes:
        gt_c = gt_by_class.get(c, [])
        pred_c = pred_by_class.get(c, [])

        if len(gt_c) == 0 and len(pred_c) == 0:
            tp_per_class[c] = 0
            fp_per_class[c] = 0
            fn_per_class[c] = 0
            continue

        matched_gt = set()
        matched_pred = set()

        for pi, pbox in enumerate(pred_c):
            best_iou = 0
            best_gi = -1
            for gi, gbox in enumerate(gt_c):
                if gi in matched_gt:
                    continue
                iou = box_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi != -1 and best_iou >= iou_threshold:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

        tp = len(matched_gt)
        fp = len(pred_c) - tp
        fn = len(gt_c) - tp

        tp_per_class[c] = tp
        fp_per_class[c] = fp
        fn_per_class[c] = fn

    return tp_per_class, fp_per_class, fn_per_class

def validate_epoch(model, yolo_model, epoch, validation_loader, calculate_loss, train_loss, config):
    map_metric = MeanAveragePrecision(class_metrics=True)
    val_loss = 0

    # Dictionaries to accumulate FP, FN, and TP counts across the whole validation
    total_tp = {}
    total_fp = {}
    total_fn = {}

    yolo_model.model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_loader):
            images = batch['image'].to(config.device)
            image_names = batch['image_name']
            ground_truths = batch['ground_truth']

            outputs = model(images)
            yolo_outputs = yolo_object(outputs, yolo_model, config.target_size, config.class_ids)

            for i in range(images.size(0)):
                predictions = yolo_outputs[i]
                gt = ground_truths[i]

                # Convert GT to mAP format
                gt_boxes = []
                gt_labels = []
                for obj in gt:
                    label, cx, cy, width, height = obj
                    x_min = cx - width / 2
                    y_min = cy - height / 2
                    x_max = cx + width / 2
                    y_max = cy + height / 2
                    gt_boxes.append([x_min, y_min, x_max, y_max])
                    gt_labels.append(label)

                gt_data = {
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "labels": torch.tensor(gt_labels, dtype=torch.int64),
                }

                # Convert predictions to mAP format
                pred_boxes = []
                pred_scores = []
                pred_labels = []
                for pred in predictions:
                    pred_boxes.append(pred['bbox'])
                    pred_scores.append(pred['confidence'])
                    pred_labels.append(pred['class_id'])

                pred_data = {
                    "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                    "scores": torch.tensor(pred_scores, dtype=torch.float32),
                    "labels": torch.tensor(pred_labels, dtype=torch.int64),
                }

                save_image_with_bboxes(
                    image_tensor=outputs[i],
                    predictions=pred_data,
                    ground_truths=gt_data,
                    epoch=epoch,
                    image_name=image_names[i],
                    experiment_name=config.experiment_name
                )

                # Update mAP metric
                map_metric.update([pred_data], [gt_data])

                # Update val_loss
                val_loss += calculate_loss(images, ground_truths, yolo_model)

                # Count FP, FN, TP for each class
                tp_per_class, fp_per_class, fn_per_class = match_predictions_to_gt(
                    pred_data["boxes"], pred_data["labels"],
                    gt_data["boxes"], gt_data["labels"],
                    iou_threshold=0.5
                )

                for c in set(list(tp_per_class.keys()) + list(fp_per_class.keys()) + list(fn_per_class.keys())):
                    total_tp[c] = total_tp.get(c, 0) + tp_per_class.get(c, 0)
                    total_fp[c] = total_fp.get(c, 0) + fp_per_class.get(c, 0)
                    total_fn[c] = total_fn.get(c, 0) + fn_per_class.get(c, 0)

        val_loss /= len(validation_loader)
        map_result = map_metric.compute()

        # Extract class-wise mAP if available, else fallback to None
        map_per_class = map_result.get("map_per_class", None)

        # We have TP, FP, FN per class. Compute precision, recall, F1, and if available mAP_class.
        # If any metric not available, default to 0.
        all_classes = set(total_tp.keys()).union(total_fp.keys()).union(total_fn.keys())
        if map_per_class is None:
            # If no map_per_class available, fill with zeros
            map_per_class = [0.0]*len(all_classes)

        # Ensure map_per_class has at least as many entries as classes we have, else pad with zeros
        if len(map_per_class) < max(all_classes)+1:
            # pad with zeros
            new_map_list = [0.0]*(max(all_classes)+1)
            for idx, val in enumerate(map_per_class):
                new_map_list[idx] = val.item() if hasattr(val, 'item') else val
            map_per_class = new_map_list

        class_metrics = []
        for class_idx in sorted(all_classes):
            tp = total_tp.get(class_idx, 0)
            fp = total_fp.get(class_idx, 0)
            fn = total_fn.get(class_idx, 0)

            precision = tp / (tp + fp + 1e-8) if (tp+fp) > 0 else 0
            recall = tp / (tp + fn + 1e-8) if (tp+fn) > 0 else 0
            f1 = (2*precision*recall/(precision+recall+1e-8)) if (precision+recall) > 0 else 0

            # map_class can be directly taken if available; if not, 0
            if isinstance(map_per_class[class_idx], torch.Tensor):
                mAP_class = map_per_class[class_idx].item()
            else:
                mAP_class = float(map_per_class[class_idx]) if map_per_class[class_idx] is not None else 0.0

            class_metrics.append({
                "Epoch": epoch,
                "ClassID": class_idx,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "mAP_class": mAP_class,
                "TP": tp,
                "FP": fp,
                "FN": fn
            })

        # Save overall metrics
        map_overall = map_result.get("map", 0.0)
        if isinstance(map_overall, torch.Tensor):
            map_overall = map_overall.item()

        save_results_to_csv(epoch, train_loss, val_loss.item(), map_overall, config.experiment_name)
        # Save class-wise metrics
        save_class_results_to_csv(class_metrics, config.experiment_name)

    yolo_model.model.train()

def save_results_to_csv(epoch, train_loss, eval_loss, mAP, experiment_name):
    csv_file_path = os.path.join(experiment_name, "results.csv")
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Train Loss", "Eval Loss", "mAP"])
        writer.writerow([epoch, train_loss, eval_loss, mAP])

def save_class_results_to_csv(class_metrics, experiment_name):
    csv_file_path = os.path.join(experiment_name, "class_results.csv")
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "ClassID", "Precision", "Recall", "F1", "mAP_class", "TP", "FP", "FN"])

        for cm in class_metrics:
            writer.writerow([
                cm["Epoch"],
                cm["ClassID"],
                cm["Precision"],
                cm["Recall"],
                cm["F1"],
                cm["mAP_class"],
                cm["TP"],
                cm["FP"],
                cm["FN"]
            ])

def save_image_with_bboxes(image_tensor, predictions, ground_truths, epoch, image_name, experiment_name):
    epoch_dir = os.path.join(experiment_name, f"Epoch{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image_np)

    height, width, _ = image_np.shape

    # GT in green
    for box, label in zip(ground_truths["boxes"], ground_truths["labels"]):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min * width, y_min * height), (x_max - x_min) * width, (y_max - y_min) * height,
            linewidth=2, edgecolor="green", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x_min * width, (y_min * height) - 5, f"GT: {label.item()}",
            color="green", fontsize=8, fontweight="bold", backgroundcolor="white"
        )

    # Predictions in red dashed
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min * width, y_min * height), (x_max - x_min) * width, (y_max - y_min) * height,
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(
            x_min * width, (y_max * height) + 5, f"Pred: {label.item()} ({score:.2f})",
            color="red", fontsize=8, fontweight="bold", backgroundcolor="white"
        )

    file_path = os.path.join(epoch_dir, f"{image_name.split('.')[0]}_Enhanced.png")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from modelparts.loss import calculate_loss
from modelparts.yolo import yolo_object
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from modelparts.loadData import ExDarkDataset, custom_collate_fn, ExDark

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

def validate_epoch_without_model(epoch, validation_loader, config):
    """
    Process validation data without using the model.
    This function will visualize ground truth bounding boxes and save the images.
    
    Args:
        epoch (int): Current epoch number.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        config (object): Configuration object containing necessary attributes.
    """
    # Initialize metrics or counters if needed (optional)
    # For example, you can calculate statistics based on ground truths

    # Create directories for saving results
    epoch_dir = os.path.join(config.experiment_name, f"Epoch{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_loader):
            images = batch['image'].to(config.device)
            image_names = batch['image_name']
            ground_truths = batch['ground_truth']

            for i in range(images.size(0)):
                gt = ground_truths[i]
                image_tensor = images[i]
                image_name = image_names[i]

                # Convert GT to appropriate format for visualization
                gt_boxes = []
                gt_labels = []
                for obj in gt:
                    label, cx, cy, width, height = obj
                    x_min = cx - width / 2
                    y_min = cy - height / 2
                    x_max = cx + width / 2
                    y_max = cy + height / 2
                    gt_boxes.append([x_min, y_min, x_max, y_max])
                    gt_labels.append(label)

                gt_data = {
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "labels": torch.tensor(gt_labels, dtype=torch.int64),
                }

                # Save image with ground truth bounding boxes
                save_image_with_bboxes(
                    image_tensor=image_tensor,
                    predictions=None,  # No predictions to display
                    ground_truths=gt_data,
                    epoch=epoch,
                    image_name=image_name,
                    experiment_name=config.experiment_name
                )

    # Optionally, save any aggregated metrics or statistics here
    # For example, count total objects per class, etc.

def save_image_with_bboxes(image_tensor, predictions, ground_truths, epoch, image_name, experiment_name):
    """
    Save the image with ground truth (and optionally predictions) bounding boxes.

    Args:
        image_tensor (torch.Tensor): Image tensor.
        predictions (dict or None): Predictions dictionary. If None, only ground truths are drawn.
        ground_truths (dict): Ground truth data.
        epoch (int): Current epoch number.
        image_name (str): Name of the image file.
        experiment_name (str): Name of the experiment for directory structure.
    """
    epoch_dir = os.path.join(experiment_name, f"Epoch{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image_np)

    height, width, _ = image_np.shape

    # Draw Ground Truths in Green
    for box, label in zip(ground_truths["boxes"], ground_truths["labels"]):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min * width, y_min * height), 
            (x_max - x_min) * width, 
            (y_max - y_min) * height,
            linewidth=2, edgecolor="green", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x_min * width, (y_min * height) - 5, f"GT: {label.item()}",
            color="green", fontsize=8, fontweight="bold", backgroundcolor="white"
        )

    # Optionally, draw Predictions if available
    if predictions is not None:
        for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min * width, y_min * height), 
                (x_max - x_min) * width, 
                (y_max - y_min) * height,
                linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(
                x_min * width, (y_max * height) + 5, f"Pred: {label.item()} ({score:.2f})",
                color="red", fontsize=8, fontweight="bold", backgroundcolor="white"
            )

    file_path = os.path.join(epoch_dir, f"{os.path.splitext(image_name)[0]}_Enhanced.png")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)



if __name__ == "__main__":
    class BaseConfig:   
        dataset_path = "DatasetExDark"
        experiment_name = "base_experiment"
        batch_size = 32
        num_workers = 8
        target_size = (640, 640)
        alt_loss_pattern = []
        class_filter = [1,2] #Bicycle(1), Boat(2), Bottle(3), Bus(4), Car(5), Cat(6), Chair(7), Cup(8), Dog(9), Motorbike(10), People(11), Table(12)
        light_filter = None                         #Low(1), Ambient(2), Object(3), Single(4), Weak(5), Strong(6), Screen(7), Window(8), Shadow(9), Twilight(10)
        location_filter = None                      #Indoor(1), Outdoor(2)
        num_epochs = 15
        learning_rate = 1e-5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = BaseConfig()
    
    dataset = ExDark(filepath=config.dataset_path)
    validation_image_paths = dataset.load_image_paths_and_classes(config, split_filter=[2])
    validation_dataset = ExDarkDataset(dataset, validation_image_paths, config.target_size)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    validate_epoch_without_model(0, validation_loader, config)