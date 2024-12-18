import numpy as np
from modelparts.loadData import ExDark

def get_image_dimensions(dataset):
    """
    Calculate the largest, smallest, and median dimensions of images in the dataset.

    Parameters:
    - dataset (ExDark): An instance of the ExDark dataset.

    Returns:
    - A dictionary with largest, smallest, and median dimensions.
    """
    image_dimensions = []

    # Load all image paths
    all_images = dataset.load_image_paths_and_classes()

    for image_name in all_images:
        image = dataset.load_image(image_name)
        if image is not None:
            # Get dimensions (width, height)
            image_dimensions.append(image.size)  # PIL Image size returns (width, height)

    if not image_dimensions:
        raise ValueError("No images loaded or dataset is empty.")

    # Convert to numpy array for easier calculation
    dims_array = np.array(image_dimensions)

    # Calculate largest, smallest, and median dimensions
    largest = dims_array.max(axis=0)
    smallest = dims_array.min(axis=0)
    median = np.median(dims_array, axis=0)

    return {
        "Largest Dimensions (Width, Height)": tuple(largest),
        "Smallest Dimensions (Width, Height)": tuple(smallest),
        "Median Dimensions (Width, Height)": tuple(map(int, median)),
    }


if __name__ == "__main__":
    dataset = ExDark(filepath="DatasetExDark")

    # Get image dimension statistics
    try:
        dimensions_stats = get_image_dimensions(dataset)
        for key, value in dimensions_stats.items():
            print(f"{key}: {value}")
    except ValueError as e:
        print(f"Error: {e}")