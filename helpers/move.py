import os
import shutil

# Define the paths to the directories
source_directory = "groundTruth"
destination_directory = "gt_test"
bundle_file_path = "test.split1.bundle"

# Read the file names from the bundle file
with open(bundle_file_path, 'r') as bundle_file:
    file_names = bundle_file.read().splitlines()

# Move each file from the source to the destination directory
for file_name in file_names:
    source_path = os.path.join(source_directory, file_name)
    destination_path = os.path.join(destination_directory, file_name)

    try:
        shutil.move(source_path, destination_path)
        print(f"Moved {file_name} to {destination_directory}")
    except FileNotFoundError:
        print(f"File {file_name} not found in {source_directory}")
    except Exception as e:
        print(f"Error moving {file_name}: {e}")