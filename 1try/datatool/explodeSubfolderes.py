import os
import shutil

def move_files_to_main_folder(main_folder_path):
    # Walk through all subfolders in the main folder
    for root, dirs, files in os.walk(main_folder_path, topdown=False):
        # For each file, move it to the main folder
        for file in files:
            file_path = os.path.join(root, file)
            shutil.move(file_path, main_folder_path)
        # Remove the subfolder after moving all its files
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            shutil.rmtree(dir_path)

    print("All files moved to the main folder, and subfolders deleted.")

# Example usage
move_files_to_main_folder('DatasetExDark/ExDark_images')