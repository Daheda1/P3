from ultralytics import YOLO
import logging

logging.getLogger('ultralytics').setLevel(logging.WARNING)


def yolo(file):
    model = YOLO("yolo11n.pt")
    results = model(file, classes=[1,8,39,5,2,15,56,41,16,3,0,60], show_labels=False)

    for result in results:
        boxes = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            xyxy = box.xyxy.squeeze().int().tolist()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = xyxy
            
            boxes.append([cls_id, x_min, y_min, x_max, y_max])

    return boxes

def yolo_object(image_part, class_ids):
    if image_part is None or image_part.size == 0:
        logging.error("Invalid image input for YOLO object detection.")
        return {}
    
    # Initialize the YOLO model with the specified weights
    model = YOLO("yolo11n.pt")
    
    # Perform inference on the input image
    results = model(image_part, conf=0.1, classes=class_ids)
    
    # Initialize a dictionary to store the cumulative confidence scores for each class ID
    confidence_scores = {cls_id: 0.0 for cls_id in class_ids}
    
    # Iterate over the detected objects
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_id in class_ids:
                    confidence_scores[cls_id] += conf
    
    # Return the cumulative confidence scores for each class ID as a dictionary
    return confidence_scores