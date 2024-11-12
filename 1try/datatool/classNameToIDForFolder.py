import os

# Mapping of class names to their numerical keys
class_name_to_id = {
    "Bicycle": 1,
    "Boat": 8,
    "Bottle": 39,
    "Bus": 5,
    "Car": 2,
    "Cat": 15,
    "Chair": 56,
    "Cup": 41,
    "Dog": 16,
    "Motorbike": 3,
    "People": 0,
    "Table": 60
}

# Function to process each file and replace class names with keys
def replace_class_names(folder_path):
    # Traverse through all subfolders and files
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(subdir, file)

                # Read the file content
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Process each line and replace class names
                new_lines = []
                for line in lines:
                    if line.startswith('%'):  # Skip header
                        new_lines.append(line)
                        continue

                    parts = line.strip().split()
                    class_name = parts[0]
                    class_id = class_name_to_id.get(class_name, "Unknown Class")

                    # Replace the class name with the ID if known
                    if class_id != "Unknown Class":
                        parts[0] = str(class_id)
                    new_lines.append(' '.join(parts))

                # Write the modified content back to the file
                with open(file_path, 'w') as f:
                    f.write('\n'.join(new_lines))

# Specify the path to the main folder
folder_path = 'DatasetExDark/ExDark_Annno'
replace_class_names(folder_path)