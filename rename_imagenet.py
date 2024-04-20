import os

# Path to the directory containing the .tar files and the mapping file
directory_path = '/cluster/scratch/vjimenez/cleanlab/train'

mapping_file_path = '/cluster/home/vjimenez/adv_pa_new/mapping_imagenet.txt'

# Read the mapping file and create a dictionary to hold the mapping
mapping_dict = {}
with open(mapping_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_id = parts[0]
            number = int(parts[1]) - 1
            mapping_dict[image_id] = number

# Rename the .tar files according to the mapping
for filename in os.listdir(directory_path):
    if filename.endswith('.tar'):
        # Extract the ID from the filename
        file_id = filename.split('.')[0]
        if file_id in mapping_dict:
            new_filename = f"{mapping_dict[file_id]}.tar"
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}")
