import os
import json

def count_annotations_in_dir(main_dir):

    """""
    Count the total number of annotations in a directory containing JSON files.
    The JSON files are assumed to have the "corrected" suffix.
    """""

    total_annotations = 0

    # Iterate through subdirectories
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        
        if os.path.isdir(subdir_path):  # Ensure it's a directory
            # Look for JSON files with "corrected" suffix
            for file in os.listdir(subdir_path):
                if file.endswith("corrected.json"):  # Adjusted for JSON
                    file_path = os.path.join(subdir_path, file)
                    
                    # Load JSON file and count annotations
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            total_annotations += sum(len(item["regions"]) for item in data.values())
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file: {file_path}")

    print(f"Total annotations in {main_dir}: {total_annotations}")

path_folder = "C:/Projects/FreeZem/NeonatesCounter/Data For Annotation/AnnotatedData\Data for annotation_Corrected/ALL"
count_annotations_in_dir(path_folder)
