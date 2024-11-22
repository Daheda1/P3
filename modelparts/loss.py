import torch
import torch.nn.functional as F
from modelparts.boxcutter import crop_bounding_boxes
from modelparts.yolo import yolo_object
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")




def create_multilabel_from_dict(results, obj):
    """
    Skaber en multilabel indikatorliste baseret på tilstedeværelsen af objekter.
    """
    indicator_list = [1 if key == obj else 0 for key in results.keys()]
    return torch.tensor(indicator_list, dtype=torch.float32, requires_grad=True)


def calculate_loss(image_tensor, ground_truth, class_ids=[1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]):
    """
    Beregner den samlede loss for et billede baseret på ground truth og YOLO-forudsigelser.
    """
    print(f"Image tensor shape: {image_tensor.shape}")


    # Split image and objects into parts
    image_parts, objects = crop_bounding_boxes(image_tensor, ground_truth)
    
    y_pred = []
    y_true = []

    for image_part, obj in zip(image_parts, objects):
        # Få YOLO-forudsigelser (sandsynligheder for hver klasse)
        results = yolo_object(image_part, class_ids)

        # Sørg for, at y_pred er sandsynligheder (output skal være mellem 0 og 1)
        probabilities = torch.tensor(list(results.values()), dtype=torch.float32)
        probabilities = torch.clamp(probabilities, 1e-10, 1 - 1e-10)  # Sikrer sandsynligheder

        y_pred.append(probabilities)
        y_true.append(create_multilabel_from_dict(results, obj))

    # Konverter lister til tensors
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    # Beregn Binary Crossentropy Loss med PyTorch
    loss = F.binary_cross_entropy(y_pred, y_true)

    return loss