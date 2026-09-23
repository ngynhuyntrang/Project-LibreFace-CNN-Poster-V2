# Author: Minh Quan Dang
# Created on: May 12, 2026

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Configurations
DATA_ROOT = "data/raf_db"
IMAGE_DIR = os.path.join(DATA_ROOT, "Image", "aligned")     
LABEL_FILE = os.path.join(DATA_ROOT, "EmoLabel", "list_patition_label.txt")

OUTPUT_ROOT = os.path.join(DATA_ROOT, "processed")             

"""
Note: The official RAF-DB dataset has already split the data into 80/20 train-test split.
What we are trying to do is leave the 20% test split as is.
And for the training data, we take 20% of it to make it into the validation set. 
As a result, we get a 64/20/16 train/test/valid split
"""
TRAIN_RATIO = 0.80
RANDOM_STATE = 69420

# Read label file
df = pd.read_csv(LABEL_FILE, sep=' ', header=None, names=['image', 'label'])
df['label'] = df['label'].astype(int)

# Fix filename: convert train_00001.jpg → train_00001_aligned.jpg
df['original_image'] = df['image'].str.replace('.jpg', '_aligned.jpg')

# Separate train and test
train_df = df[df['image'].str.startswith('train_')].copy()
test_df = df[df['image'].str.startswith('test_')].copy()

print(f"Original Train images: {len(train_df)}")
print(f"Original Test images : {len(test_df)}")

# Split the training set
new_train_idx, val_idx = train_test_split(
    train_df.index,
    test_size=(1 - TRAIN_RATIO),
    stratify=train_df['label'],
    random_state=RANDOM_STATE
)

new_train_df = train_df.loc[new_train_idx].reset_index(drop=True)
val_df = train_df.loc[val_idx].reset_index(drop=True)

print(f"New Train set     : {len(new_train_df)} images")
print(f"Validation set    : {len(val_df)} images")

# Create directories (1-7)
for split in ['train', 'val', 'test']:
    for label in range(1, 8):
        os.makedirs(os.path.join(OUTPUT_ROOT, split, str(label)), exist_ok=True)

def copy_with_rename(df_subset, split_name):
    count = 0
    for _, row in df_subset.iterrows():
        src_path = os.path.join(IMAGE_DIR, row['original_image'])
        
        # Rename for validation set
        if split_name == 'val':
            new_name = row['image'].replace("train_", "valid_")
        else:
            new_name = row['image']
        
        label_folder = str(row['label'])
        dst_path = os.path.join(OUTPUT_ROOT, split_name, label_folder, new_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            count += 1
        else:
            print(f"Warning: Missing file {row['original_image']}")
    return count

print("\nCopying and renaming images")

train_count = copy_with_rename(new_train_df, 'train')
val_count   = copy_with_rename(val_df, 'val')
test_count  = copy_with_rename(test_df, 'test')

print("\n")
print("RAF-DB Split Creation Completed!")
print("\n")
print(f"Train set      : {train_count} images")
print(f"Validation set : {val_count} images (valid_xxxx.jpg)")
print(f"Test set       : {test_count} images")
print("\n")