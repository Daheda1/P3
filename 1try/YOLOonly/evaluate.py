import os
import json
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datatool.loaddataset import ExDark

def evaluate_model_on_exdark(model, dataset, split_filter=None):
    """
    Evaluates a YOLO model on the ExDark dataset using COCO annotations and computes mAP.

    Parameters:
    - model: YOLO model instance.
    - dataset: ExDark dataset instance.
    - split_filter (list of int, optional): Data splits to include (1: train, 2: val, 3: test).

    Returns:
    - results (dict): Dictionary containing various mAP metrics.
    """
    # Load COCO-formatted annotations using ExDark
    coco_annotations = dataset.load_annotations_coco(split_filter=split_filter)
    
    # Save combined annotations to a temporary JSON file
    temp_annotation_file = "temp_exdark_coco_annotations.json"
    with open(temp_annotation_file, 'w') as f:
        json.dump(coco_annotations, f)
    
    # Initialize COCO API for ground truth
    coco_gt = COCO(temp_annotation_file)
    
    # Get all image IDs
    img_ids = coco_gt.getImgIds()
    
    coco_dt = []
    
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(dataset.image_path, img_info['file_name'])
        
        # Ensure the image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping.")
            continue
        
        # Predict with YOLO
        results = model(image_path)
        
        for result in results:
            for box in result.boxes:
                pred_class_id = int(box.cls[0].item())
                pred_conf = box.conf[0].item()
                pred_bbox = box.xywh[0].tolist()  # [x_center, y_center, width, height]
                
                x_center, y_center, width, height = pred_bbox
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                # COCO expects [x_min, y_min, width, height]
                coco_dt.append({
                    "image_id": img_id,
                    "category_id": pred_class_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": pred_conf
                })
    
    # Remove the temporary annotation file
    os.remove(temp_annotation_file)
    
    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(coco_dt)
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract mAP metrics from COCOeval
    results = {
        "mAP_50_95": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],
        "AR_10": coco_eval.stats[7],
        "AR_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11]
    }
    
    # Print mAP metrics
    print("\nDetailed mAP Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results

# Example of running the function
if __name__ == "__main__":
    # Initialize ExDark dataset
    dataset = ExDark(filepath="DatasetExDark")
    
    # Initialize YOLO model
    model = YOLO("yolo11n.pt")
    
    # Define split filters as needed
    split_filter = [1, 2, 3]  # For example, evaluate on train, val, and test splits
    
    # Evaluate the model
    evaluation_results = evaluate_model_on_exdark(
        model=model,
        dataset=dataset,
        split_filter=split_filter
    )
    
    # Optionally, save the results to a JSON file for later analysis
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)