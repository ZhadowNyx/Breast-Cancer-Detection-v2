import os
import shutil
import random
from tqdm import tqdm

original_dir = r"E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_histology_slides"
output_dir = r"E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split"
categories = ['benign', 'malignant']

# Ratios
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Ensure output folders exist
for split in ['train', 'val', 'test']:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Process each category
for category in categories:
    source_folder = os.path.join(original_dir, category)
    images = os.listdir(source_folder)
    random.shuffle(images)

    total = len(images)
    train_end = int(train_split * total)
    val_end = train_end + int(val_split * total)

    split_data = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, file_list in split_data.items():
        for file in tqdm(file_list, desc=f"Copying {category} to {split}"):
            src_path = os.path.join(source_folder, file)
            dst_path = os.path.join(output_dir, split, category, file)
            shutil.copyfile(src_path, dst_path)