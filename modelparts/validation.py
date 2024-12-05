import csv
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader
from modelparts.loadData import ExDarkDataset, custom_collate_fn  # Ensure correct import based on your project structure
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

yolo = YOLO("yolo11n.pt")


def validate_epoch(model, epoch, device, validation_loader, calculate_loss, average_loss):

    for batch_idx, batch in enumerate(validation_loader):
        groud_truths =  batch['ground_truth']
        images = batch['image'].to(device)

        results = model(images)

    cls_id = int(box.cls.item())  # Class ID
    xyxy = box.xyxy.squeeze().int().tolist()
        
    print(results)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dataset = ExDarkDataset(csv_file="DatasetExDark/ExDark_Annno/ExDark_Annno.csv", transform=None)