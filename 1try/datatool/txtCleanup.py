import os

def clean_and_rename_text_files(folder_path):
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Clean the file by removing empty lines and lines starting with '%'
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                cleaned_lines = [line for line in lines if line.strip() and not line.strip().startswith('%')]
            
            # Write the cleaned lines back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(cleaned_lines)

            # Rename the file if it contains '.png' in the name
            if ".jpeg" in filename:
                new_filename = filename.replace(".jpeg", "")
                new_file_path = os.path.join(folder_path, new_filename)
                os.rename(file_path, new_file_path)
                
    print(f"Cleaning and renaming complete for folder: {folder_path}")

# Usage
folder_path = "DatasetExDark/ExDark_Annno"
clean_and_rename_text_files(folder_path)