from pathlib import Path
from shutil import copy


cwd = Path.cwd()
data_dir = cwd.parent.parent / 'data' / 'stanford-car-dataset-by-classes-folder'
original_train_dir = data_dir / 'car_data' / 'train'
original_test_dir = data_dir / 'car_data' / 'test'
new_data_dir = data_dir / 'car_data' / 'new_data_cleaned'
new_train_dir = data_dir / 'car_data_new_data_in_train' / 'train'
new_test_dir = data_dir / 'car_data_new_data_in_train' / 'test'


if __name__ == '__main__':
    for folder in new_data_dir.iterdir():
        if folder.is_dir():
            for image in folder.glob('*.jpg'):
                copy(str(image), str(new_train_dir / folder.stem / image.name))