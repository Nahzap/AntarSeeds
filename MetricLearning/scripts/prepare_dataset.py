"""
Script to prepare the microscopy dataset for metric learning training.
Creates symbolic links to organize images into train/val/test splits without duplicating files.
"""

import os
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

# Source directories
SOURCES = {
    'Quillaja_Saponaria': r'f:\MICROSCOPIA\Quillaja_Saponaria\ATTMPT05',
    'Medicago_sativa': r'f:\MICROSCOPIA\Medicago_sativa\ATTMPT_06',
    'Lithraea_caustica': r'f:\MICROSCOPIA\Lithraea_caustica\ATTMPT_01'
}

# Target directories
BASE_DIR = Path(r'c:\Users\askna\Documents\GitHub\MetricLearning')
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_directory_structure():
    """Create the necessary directory structure"""
    for split in ['train', 'val', 'test']:
        for class_name in SOURCES.keys():
            target_dir = PROCESSED_DIR / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {target_dir}")

def get_image_files(source_dir):
    """Get all PNG image files from source directory"""
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Warning: {source_dir} does not exist!")
        return []
    
    images = list(source_path.glob('*.png'))
    print(f"Found {len(images)} images in {source_dir}")
    return images

def split_and_link_images(class_name, source_dir):
    """Split images into train/val/test and create symbolic links"""
    images = get_image_files(source_dir)
    
    if len(images) == 0:
        print(f"No images found for {class_name}, skipping...")
        return
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    # Split images
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    print(f"\n{class_name}:")
    print(f"  Total: {n_total}")
    print(f"  Train: {len(train_images)}")
    print(f"  Val: {len(val_images)}")
    print(f"  Test: {len(test_images)}")
    
    # Create symbolic links to images in respective directories
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, image_list in splits.items():
        target_dir = PROCESSED_DIR / split_name / class_name
        for img_path in image_list:
            target_path = target_dir / img_path.name
            # Remove existing symlink if it exists
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            # Create symbolic link
            try:
                target_path.symlink_to(img_path)
            except OSError:
                # If symlink fails (permissions), fall back to hardlink
                try:
                    os.link(img_path, target_path)
                except:
                    print(f"    Warning: Could not link {img_path.name}, copying instead")
                    shutil.copy2(img_path, target_path)
    
    print(f"  ✓ Created links for {class_name}")

def main():
    print("="*70)
    print("PREPARING MICROSCOPY DATASET FOR METRIC LEARNING")
    print("="*70)
    
    # Create directory structure
    print("\nCreating directory structure...")
    create_directory_structure()
    
    # Process each class
    print("\n" + "="*70)
    print("SPLITTING AND COPYING IMAGES")
    print("="*70)
    
    for class_name, source_dir in SOURCES.items():
        split_and_link_images(class_name, source_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE")
    print("="*70)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = PROCESSED_DIR / split
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                n_images = len(list(class_dir.glob('*.png')))
                print(f"{split}/{class_dir.name}: {n_images} images")
                if split == 'train':
                    total_train += n_images
                elif split == 'val':
                    total_val += n_images
                else:
                    total_test += n_images
    
    print(f"\nTotal - Train: {total_train}, Val: {total_val}, Test: {total_test}")
    print(f"Grand Total: {total_train + total_val + total_test}")
    print("\n✓ Dataset ready for training!")

if __name__ == '__main__':
    main()
