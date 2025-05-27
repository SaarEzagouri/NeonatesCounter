import json
"""
This script merges and updates image annotation entries in a main JSON file using replacement data from several other JSON files.
The script is designed for use with VGG (VIA) annotation JSONs, where each annotation entry is keyed by a unique identifier and contains at least a "filename" field. It replaces all annotation entries in the main JSON whose filenames match a specified set, using updated entries from the provided list of other JSON files. The merged result is saved to a new output JSON file.
Configuration:
    - `main_json_path` (str): Path to the main annotation JSON file to be updated.
    - `other_json_paths` (List[str]): List of paths to JSON files containing updated annotation entries.
    - `filenames_to_replace` (Set[str]): Set of filenames whose annotations should be replaced in the main JSON.
    - `output_json_path` (str): Path to save the merged output JSON.
Workflow:
    1. Loads the main annotation JSON.
    2. Iterates through each additional JSON file:
        - For each entry, if its filename is in `filenames_to_replace`, removes all existing entries in the main JSON with that filename.
        - Adds the new annotation entry from the additional JSON.
    3. Saves the merged annotations to the specified output path.
Prints:
    - Logs each removal and addition of annotation entries.
    - Prints a completion message with the output file path.
Raises:
    - FileNotFoundError: If any of the specified JSON files do not exist.
    - json.JSONDecodeError: If any JSON file is malformed.
    - OSError: If there are issues creating the output directory or writing the output file.
"""
import os

# === CONFIGURATION ===
main_json_path = "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/neonates_counter_json_final.json"
other_json_paths = [
    "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/250512_110123.json",
    "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/250512_132122.json",
    "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/250512_094339.json",
    "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/250512_132539.json",
    "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/250512_131736.json",
    "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/250512_114258.json",
]
filenames_to_replace = {
    "250512_110123.jpg",
    "250512_132122.jpg",
    "250512_094339.jpg",#
    "250512_132539.jpg",#
    "250512_131736.jpg",#
    "250512_114258.jpg", #
}
output_json_path = "C:/Projects/FreeZem/NeonatesCounter/Annotations_eval/neonates_counter_json_merged.json"

# === LOAD MAIN JSON ===
with open(main_json_path, "r") as f:
    main_data = json.load(f)

# === REPLACE ANNOTATIONS BASED ON FILENAMES ===
for json_file in other_json_paths:
    with open(json_file, "r") as f:
        data = json.load(f)
        for key, entry in data.items():
            fname = entry.get("filename", "")
            if fname in filenames_to_replace:
                # Remove ALL existing entries with this filename (regardless of key)
                keys_to_remove = [k for k, v in main_data.items() if v.get("filename") == fname]
                for k in keys_to_remove:
                    print(f"Removing old annotation for '{fname}' (key: {k})")
                    del main_data[k]
                # Add the new annotation
                print(f"Adding new annotation for '{fname}' (key: {key})")
                main_data[key] = entry

# === SAVE MERGED OUTPUT ===
output_dir = os.path.dirname(output_json_path)
os.makedirs(output_dir, exist_ok=True)

with open(output_json_path, "w") as f:
    json.dump(main_data, f, indent=2)

print(f"\n✅ Done. Merged annotations saved to:\n{output_json_path}")
