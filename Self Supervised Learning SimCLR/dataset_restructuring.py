import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Source and destination paths
source_dir = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\unlabeled'
dest_base = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\unlabeled_split'

# Create split folders
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(dest_base, split), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)

# Split indices
total = len(image_files)
train_end = int(0.7 * total)
val_end = train_end + int(0.15 * total)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

# Helper to copy files
def copy_files(file_list, split_name):
    for fname in file_list:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_base, split_name, fname)
        shutil.copy2(src, dst)

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print(f"✅ Done! Split {total} images into:")
print(f"→ Train: {len(train_files)}")
print(f"→ Val: {len(val_files)}")
print(f"→ Test: {len(test_files)}")
