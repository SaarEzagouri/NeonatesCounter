import os
import shutil

def copy_images_by_folder(input_folder, train_folders=None, val_folders=None, test_folders=None, path1=None, path2=None, path3=None):
    """
    This function copies JPEG images from subfolders to specified directories based on folder names.
    
    Args:
    - input_folder (str): The root folder containing subfolders of images.
    - train_folders (list): List of folder names to copy images to path1.
    - val_folders (list): List of folder names to copy images to path2.
    - test_folders (list): List of folder names to copy images to path3.
    - path1 (str or None): Destination directory for 'train' images. If None, train images won't be copied.
    - path2 (str or None): Destination directory for 'val' images. If None, val images won't be copied.
    - path3 (str or None): Destination directory for 'test' images. If None, test images won't be copied.
    """
    # Ensure the destination paths exist if provided
    if path1:
        os.makedirs(path1, exist_ok=True)
    if path2:
        os.makedirs(path2, exist_ok=True)
    if path3:
        os.makedirs(path3, exist_ok=True)

    # Loop over the subfolders in the input folder
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            dest_path = None  # Initialize destination path
            
            # Determine the destination path based on the folder name
            if folder_name in train_folders and path1:
                dest_path = path1
            elif folder_name in val_folders and path2:
                dest_path = path2
            elif folder_name in test_folders and path3:
                dest_path = path3
            
            if not dest_path:
                continue  # Skip if no valid destination path
            
            jpg_found = False

            # Loop through the files in the folder and copy JPEG files
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.jpg'):
                    source_file = os.path.join(folder_path, file_name)
                    shutil.copy(source_file, dest_path)
                    print(f"Copied {file_name} to {dest_path}")
                    jpg_found = True

            if not jpg_found:
                print(f"No JPEG files found in {folder_path}")

    print("Images were copied!")

# Example Usage
path1 = 'C:/Projects/FreeZem/NeonatesCounter/dataset/train/images'
path2 = 'C:/Projects/FreeZem/NeonatesCounter/dataset/val/images'
path3 = None  # Skipping test folder

train_folders = ['09_13_1222', '12_18_1223', '240101_105044']
val_folders = ['230614_135503']
test_folders = None

input_folder = 'C:/Projects/FreeZem/NeonatesCounter/Data For Annotation/AnnotatedData/Data for annotation_Corrected/addition/'

copy_images_by_folder(input_folder, train_folders, val_folders, test_folders, path1, path2, path3)
