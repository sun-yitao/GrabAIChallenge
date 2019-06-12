import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import random

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from lib.random_eraser import get_random_eraser
from lib import Automold as am


IMAGE_SIZE = (363, 525)  # height, width, avg is (483,700) (525,766)
N_CLASSES = 196
LR_FINAL = 0.01
BATCH_SIZE = 64

cwd = Path.cwd()
DATA_DIR = cwd.parent / 'data'
TRAIN_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / \
    'car_data_new_data_in_train_v2' / 'train'
SAVE_DIR = DATA_DIR / 'augment_samples'
os.makedirs(SAVE_DIR, exist_ok=True)


def augment_np_image(image):
    if random.random() > 0.7:
        image = am.augment_random(image, aug_types=['add_rain', 'add_sun_flare'], volume='same')
        image = image.astype(np.float32)
    eraser = get_random_eraser(p=0.8, s_l=0.02, s_h=0.3, r_1=0.3, r_2=1/0.3,
                               v_l=0, v_h=255, pixel_level=True)
    image = eraser(image)
    return image


def get_input_data_generator():
    # preprocessing function executes before rescale
    train_datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0.2,
                                       height_shift_range=0.2, brightness_range=(0.8, 1.2),
                                       shear_range=0.1, zoom_range=0.2,
                                       channel_shift_range=0.2,
                                       fill_mode='reflect', horizontal_flip=True,
                                       vertical_flip=False, rescale=1/255,
                                       preprocessing_function=augment_np_image)
    train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE,
                                              save_to_dir=str(SAVE_DIR), save_format='jpg',
                                              color_mode='rgb', batch_size=BATCH_SIZE, interpolation='lanczos')
    return train


if __name__ == '__main__':
    train = get_input_data_generator()
    train.next()