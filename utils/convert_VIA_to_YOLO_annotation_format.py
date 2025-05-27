from YOLO_VIA_Annotations import from_via_anno_to_txt_bbox
import os

"""""
Output a directory called 'labels' in the same directory as the input json file.
"""""

if __name__ == '__main__':
    # Get the base directory where all subfolders are stored
    base_dir = "C:/Projects/FreeZem/NeonatesCounter/Data For Annotation/AnnotatedData/Data for annotation_Corrected/addition/"  # Change this to the actual path of your dataset

    # Loop through all subdirectories and find annotation.json files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "annotations_corrected.json":  # Adjust this if your annotation files have a different name
                json_path = os.path.join(root, file)
                print(f"Processing: {json_path}")
                from_via_anno_to_txt_bbox(json_path, type_str=".json",parent_n=2)

    print("Conversion complete!")