import os
import json
from PIL import Image
import numpy as np
import cv2

class ExDark:
    def __init__(self, filepath):
        self.annotation_path = os.path.join(filepath, "ExDark_Annno")  # Corrected path
        self.image_path = os.path.join(filepath, "ExDark_images")
        self.image_list = os.path.join(filepath, "imageclasslist.txt")
        self.class_map = self.get_class_map()
        # Create a reverse map for class names to IDs
        self.class_name_to_id = {v: k for k, v in self.class_map.items()}
        # Define the path for COCO JSON files (assumed to be in the root of filepath)
        self.coco_annotations_path = filepath  # Adjust if JSONs are in a different directory

    def load_image_paths_and_classes(self, class_filter=None, light_filter=None, location_filter=None, split_filter=None):
        """
        Loads image paths and classes with optional filters.
        
        Parameters:
        - class_filter (list of int): List of class numbers to include.
        - light_filter (list of int): List of light condition numbers to include.
        - location_filter (list of int): List of location types (1 for indoor, 2 for outdoor) to include.
        - split_filter (list of int): List of data splits (1 for train, 2 for val, 3 for test) to include.

        Returns:
        - List of tuples: (image_path, class_id)
        """
        with open(self.image_list, 'r') as file:
            lines = file.readlines()
        
        image_paths = []
        image_classes = []
        
        for line in lines[1:]:  # Skip header line
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Warning: Malformed line in image list: {line}")
                continue
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
                
                class_name = self.get_class_name(img_class)
                if not class_name:
                    print(f"Warning: Unknown class ID {img_class} for image {image_name}")
                    continue
                
                image_path = os.path.join(self.image_path, class_name, image_name)
                image_paths.append(image_path)
                image_classes.append(img_class)
        
        return list(zip(image_paths, image_classes))

    def load_annotations_coco(self, split_filter=None):
        """
        Loads annotations in COCO format from train.json, val.json, and test.json based on split_filter.

        Parameters:
        - split_filter (list of int): List of split numbers to include (1: train, 2: val, 3: test).
                                        If None, all splits are loaded.

        Returns:
        - dict: Combined COCO annotations containing 'images', 'annotations', and 'categories'.
        """
        split_map = {1: 'train.json', 2: 'val.json', 3: 'test.json'}
        combined_annotations = {
            'images': [],
            'annotations': [],
            'categories': self.get_coco_categories()
        }

        # Determine which splits to load
        splits_to_load = split_filter if split_filter is not None else list(split_map.keys())

        annotation_id = 1  # Initialize annotation ID for COCO

        # Load image paths and classes for mapping
        image_class_mapping = {}
        image_class_list = self.load_image_paths_and_classes(split_filter=split_filter)
        for image_path, class_id in image_class_list:
            filename = os.path.basename(image_path)
            image_class_mapping[filename] = self.get_class_name(class_id)

        for split_num in splits_to_load:
            split_file = os.path.join(self.coco_annotations_path, split_map.get(split_num))
            if not split_file or not os.path.exists(split_file):
                print(f"Warning: Split file for split {split_num} not found at {split_file}. Skipping.")
                continue

            try:
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                
                # Adjust annotation IDs to ensure uniqueness
                for ann in split_data.get('annotations', []):
                    ann['id'] = annotation_id
                    annotation_id += 1

                # Modify 'file_name' to include class subfolder
                for image in split_data.get('images', []):
                    filename = image['file_name']
                    class_name = image_class_mapping.get(filename)
                    if class_name:
                        image['file_name'] = os.path.join(class_name, filename)
                    else:
                        print(f"Warning: No class found for image {filename}. Using original file name.")
                
                combined_annotations['images'].extend(split_data.get('images', []))
                combined_annotations['annotations'].extend(split_data.get('annotations', []))
                # Assuming 'categories' are consistent across splits; otherwise, handle accordingly

                print(f"Loaded {len(split_data.get('images', []))} images and {len(split_data.get('annotations', []))} annotations from {split_map[split_num]}")
            
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {split_file}: {e}")
            except Exception as e:
                print(f"Unexpected error while loading {split_file}: {e}")

        return combined_annotations

    def get_class_name(self, class_id):
        """Maps class ID to class name as per the dataset structure."""
        return self.class_map.get(class_id, None)
    
    def get_class_map(self):
        return {
            1: "Bicycle",
            2: "Boat",
            3: "Bottle",
            4: "Bus",
            5: "Car",
            6: "Cat",
            7: "Chair",
            8: "Cup",
            9: "Dog",
            10: "Motorbike",
            11: "People",
            12: "Table"
        }
    

# Test example
if __name__ == "__main__":
    dataset = ExDark(filepath="DatasetExDark")
    
    # Example usage of existing methods
    filtered_images = dataset.load_image_paths_and_classes(class_filter=[1], split_filter=[2])
    print("Filtered Images:", filtered_images)
    
    image = dataset.load_image("2015_00001.png", 1)
    print("Loaded Image:", image)
    
    annotations = dataset.load_annotations("2015_00001.png", 1)
    print("Annotations for Image:", annotations)
    
    # Example usage of the new load_annotations_coco method
    coco_annotations = dataset.load_annotations_coco(split_filter=[1, 2])  # Load train and val splits
    print("COCO Annotations:", coco_annotations)