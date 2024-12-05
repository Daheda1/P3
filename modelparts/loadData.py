import os
import cv2 as cv
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from modelparts.imagePreprocessing import scale_image, scale_bounding_boxes, pad_image_to_target, convert_to_ultralytics_format
import torchvision.transforms.functional as TF




class ExDark:
    def __init__(self, filepath):
        self.annotations = os.path.join(filepath, "ExDark_Anno/")
        self.images = os.path.join(filepath, "ExDark_images/")
        self.image_list = os.path.join(filepath, "imageclasslist.txt")

    def load_image_paths_and_classes(self, class_filter=None, light_filter=None, location_filter=None, split_filter=None):
        """
        Loads image paths and classes with filters.
        
        Parameters:
        - class_filter (list of int): List of class numbers to include.
        - light_filter (list of int): List of light condition numbers to include.
        - location_filter (list of int): List of location types (1 for indoor, 2 for outdoor) to include.
        - split_filter (list of int): List of data splits (1 for train, 2 for val, 3 for test) to include.
        """
        with open(self.image_list, 'r') as file:
            lines = file.readlines()
        
        image_paths = []
        image_classes = []
        
        for line in lines[1:]:  # Skip header line
            parts = line.split()
            image_name = parts[0]
            img_class = int(parts[1])
            light_condition = int(parts[2])
            location = int(parts[3])
            split = int(parts[4])
            
            # Apply filters
            if (class_filter is None or img_class in class_filter) and \
               (light_filter is None or light_condition in light_filter) and \
               (location_filter is None or location in location_filter) and \
               (split_filter is None or split in split_filter):
                image_paths.append(os.path.join(image_name))
                image_classes.append(img_class)
        
        return list(image_paths)
    
    def load_ground_truth(self, image_name):
        """
        Loads annotations for a given image and returns them as a list of lists.
        
        Parameters:
        - image_name (str): The name of the image file.
        
        Returns:
        - List of lists, where each inner list represents an annotation with [class, x1, y1, x2, y2].
        """
        base = os.path.splitext(image_name)[0]  # Remove current extension
        annotation_path = os.path.join(self.annotations, base + ".txt")  # Correctly form the .txt path

        ground_truth = []
        
        # Check if the annotation file exists
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as file:
                for line in file:
                    parts = line.split()
                    # Parse each line and take the first 5 elements
                    # Converting each to integer
                    annotation = list(map(int, parts[:5]))
                    ground_truth.append(annotation)
        
        return ground_truth
        
    def load_image(self, image_name):
        """
        Loads an image by its name, handling errors and adding logging.
        
        Parameters:
        - image_name (str): The name of the image file.
        
        Returns:
        - Loaded image (numpy array) if successful, or None if loading failed.
        """
        image_path = os.path.join(self.images, image_name)
        
        if not os.path.exists(image_path):
            logging.error(f"Image file does not exist: {image_path}")
            return None
        
        image = cv.imread(image_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        
        if image is None or image.size == 0:
            logging.error(f"Failed to load image or image is empty: {image_path}")
            return None
        
        #logging.info(f"Image loaded successfully: {image_path}, Shape: {image.shape}")
        return image


if __name__ == "__main__":
    dataset = ExDark(filepath="DatasetExDark")
    
    # Example usage of existing methods
    filtered_images = dataset.load_image_paths_and_classes(class_filter=[1], split_filter=[2])
    print("Filtered Images:", filtered_images)

    # Test loading an image
    if filtered_images:
        for image_name in filtered_images:
            image = dataset.load_image(image_name)
            if image is not None:
                print(f"Image {image_name} loaded successfully.")


class ExDarkDataset(Dataset):
    def __init__(self, dataset, image_paths, target_size=(640, 640), transform=None, divideable_by=32):
        self.dataset = dataset
        self.image_paths = image_paths
        self.target_size = target_size  # (width, height)
        self.transform = transform
        self.divideable_by = divideable_by

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image = self.dataset.load_image(image_name)
        ground_truth = self.dataset.load_ground_truth(image_name)

        image, resize_ratio = scale_image(image, self.target_size, self.divideable_by)
        image, padding = pad_image_to_target(image, self.target_size)
        ground_truth = scale_bounding_boxes(ground_truth, resize_ratio, padding)
        ground_truth = convert_to_ultralytics_format(ground_truth, self.target_size)

        # Ensure the final image size is as expected
        assert image.size == (self.target_size[0], self.target_size[1]), (
            f"Image size after padding is {image.size}, expected {self.target_size}"
        )

        # Convert image to tensor
        image_tensor = TF.to_tensor(image)  # Automatically scales to [0,1]

        sample = {
            'image': image_tensor,
            'Org_image': image_tensor,
            'ground_truth': ground_truth,
            'padding': padding,
            'image_name': image_name  # Maybe not needed
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    



def custom_collate_fn(batch):
    batch_element = batch[0]
    collated_batch = {}
    for key in batch_element:
        if key in ['padding', 'ground_truth', 'image_name']:
            # Keep these as lists without collating into tensors
            collated_batch[key] = [d[key] for d in batch]
        else:
            # Use default collation for other fields
            collated_batch[key] = torch.stack([d[key] for d in batch], dim=0)
    return collated_batch