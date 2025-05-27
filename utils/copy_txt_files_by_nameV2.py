import os
import shutil

def copy_txt_files_by_name(input_folder, train_list=None, val_list=None, test_list=None, path1=None, path2=None, path3=None):
    """
    This function copies .txt files from a folder to specific directories based on the prefix of the file name.
    
    Args:
    - input_folder (str): The folder containing .txt files to be matched.
    - train_list (list or None): List of prefixes for 'train' files. If None, train files won't be processed.
    - val_list (list or None): List of prefixes for 'val' files. If None, val files won't be processed.
    - test_list (list or None): List of prefixes for 'test' files. If None, test files won't be processed.
    - path1 (str or None): Destination directory for 'train' files. If None, train files won't be copied.
    - path2 (str or None): Destination directory for 'val' files. If None, val files won't be copied.
    - path3 (str or None): Destination directory for 'test' files. If None, test files won't be copied.
    """
    # Ensure lists are not None
    train_list = train_list or []
    val_list = val_list or []
    test_list = test_list or []

    # Ensure the destination paths exist if provided
    if path1:
        os.makedirs(path1, exist_ok=True)
    if path2:
        os.makedirs(path2, exist_ok=True)
    if path3:
        os.makedirs(path3, exist_ok=True)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        # Check if the file is a .txt file
        if os.path.isfile(file_path) and file_name.lower().endswith('.txt'):
            dest_path = None  # Initialize destination path
            
            # Determine the destination path based on the file prefix
            if any(file_name.startswith(prefix) for prefix in train_list) and path1:
                dest_path = path1
            elif any(file_name.startswith(prefix) for prefix in val_list) and path2:
                dest_path = path2
            elif any(file_name.startswith(prefix) for prefix in test_list) and path3:
                dest_path = path3

            if not dest_path:
                continue  # Skip if no valid destination path
            
            # Copy the file to the corresponding destination
            shutil.copy(file_path, dest_path)
            print(f"Copied {file_name} to {dest_path}")
    
    print("Files were copied!")

# Example Usage
path1 = 'C:/Projects/FreeZem/NeonatesCounter/dataset/train/labels'
path2 = 'C:/Projects/FreeZem/NeonatesCounter/dataset/val/labels'
path3 = None  # Skipping test folder

train_list = ['2023_07_11_09_12_18_1223','2023_07_11_09_09_13_1222',
              '240101_105044']
val_list = ['230614_135503']
test_list = None

input_folder = 'C:/Projects\FreeZem/NeonatesCounter/Data For Annotation/AnnotatedData/Data for annotation_Corrected/labels/'

copy_txt_files_by_name(input_folder, train_list, val_list, test_list, path1, path2, path3)
