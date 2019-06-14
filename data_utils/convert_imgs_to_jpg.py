"""Script to convert non-jpgs in a folder to jpg"""
import os
from pathlib import Path
from PIL import Image
from glob import glob
from tqdm import tqdm
from shutil import move


cwd = Path.cwd()
DATA_DIR = cwd.parent / 'data' / 'stanford-car-dataset-by-classes-folder' / 'car_data' / 'new_data'


def convert_png_to_jpg():
    for image in glob(str(DATA_DIR / '**' / '*.png'), recursive=True):
        image = Path(image)
        print(image)
        try:
            im = Image.open(str(image))
            rgb_im = im.convert('RGB')
            rgb_im.save(str(image).replace('png', 'jpg'))
        except:
            print(f'Failed: {image}')
            continue


def convert_non_jpg_to_jpg():
    for image in tqdm(glob(str(DATA_DIR / '**' / '*'), recursive=True)):
        image = Path(image)
        if image.suffix != '.jpg':
            try:
                im = Image.open(str(image))
                rgb_im = im.convert('RGB')
                rgb_im.save(str(image).replace(image.suffix, '.jpg'))
                print('Converted')
            except Exception as e:
                print(e)
                print(f'Failed: {image}')
                continue


if __name__ == '__main__':
    convert_png_to_jpg()