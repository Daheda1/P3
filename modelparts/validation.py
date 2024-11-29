from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import torch



def validation(model, dataset, output_csv):
    image_paths = dataset.load_image_paths_and_classes(split_filter=[3])