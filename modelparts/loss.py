import numpy as np
from modelparts.boxcutter import crop_bounding_boxes
from modelparts.yolo import yolo_object
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
import torch


def create_multilabel_from_dict(results, obj):
    """
    Skaber en multilabel indikatorliste baseret på tilstedeværelsen af objekter.
    """
    indicator_list = [1 if key == obj else 0 for key in results.keys()]
    return indicator_list


def binary_crossentropy_loss(y_true, y_pred):
    """
    Beregner Binary Crossentropy Loss.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # For at undgå log(0), tilføjes en lille epsilon-værdi
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Binary Crossentropy beregning
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def calculate_loss(image, ground_truth, class_ids=[1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]):
    """
    Beregner den samlede loss for et billede baseret på ground truth og YOLO-forudsigelser.
    """
    image_parts, objects = crop_bounding_boxes(image, ground_truth)
    y_pred = []
    y_true = []

    for image_part, object in zip(image_parts, objects):
        # Få YOLO-forudsigelser (sandsynligheder for hver klasse)
        results = yolo_object(image_part, class_ids)

        # Sørg for, at y_pred er sandsynligheder (output skal være mellem 0 og 1)
        probabilities = np.array(list(results.values()))
        probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)  # Sikrer sandsynligheder

        y_pred.extend(probabilities)
        y_true.extend(create_multilabel_from_dict(results, object))

    print(y_true, y_pred)

    # Beregn loss
    loss = binary_crossentropy_loss(y_true, y_pred)
    return torch.tensor(loss, requires_grad=True)
 