import os
import random
from pathlib import Path
from shutil import move
from glob import glob


SEED = 448
cwd = Path.cwd()
data_dir = cwd.parent.parent / 'data' / 'stanford-car-dataset-by-classes-folder' / 'car_data'
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'


def combine_original_train_test_folders():
    for class_folder in test_dir.glob('*'):
        if class_folder.is_dir():
            for n, image_path in enumerate(class_folder.glob('*.jpg')):
                try:
                    move(str(image_path), str(train_dir / class_folder.name))
                except:
                    rename = str(image_path.stem) + f'_{n}' + str(image_path.suffix)
                    move(str(image_path), str(
                        train_dir / class_folder.name / rename))


def stratified_train_test_split(test_ratio=0.25):
    for class_folder in train_dir.glob('*'):
        if class_folder.is_dir():
            class_image_paths = list(class_folder.glob('*.jpg'))
            random.shuffle(class_image_paths)
            split_idx = int(len(class_image_paths) * test_ratio)
            test_image_paths = class_image_paths[:split_idx]
            for image_path in test_image_paths:
                move(str(image_path), str(test_dir / class_folder.name))


if __name__ == '__main__':
    random.seed(SEED)
    original_num_images = len(list(
        glob(str(data_dir / '**' / '*.jpg'), recursive=True)
    ))
    combine_original_train_test_folders()
    stratified_train_test_split()
    final_num_images = len(list(
        glob(str(data_dir / '**' / '*.jpg'), recursive=True)
    ))
    assert original_num_images == final_num_images
