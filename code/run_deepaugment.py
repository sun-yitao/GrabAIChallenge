import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from deepaugment.deepaugment import DeepAugment

cwd = Path.cwd()
DATA_DIR = cwd.parent / 'data'
TRAIN_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / 'car_data' / 'train'
TEST_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / 'car_data' / 'test'
IMAGE_SIZE = (363, 525)
TRAIN_SET_SIZE = 2000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

my_config = {
    "model": "basiccnn",
    'train_set_size': int(TRAIN_SET_SIZE*0.75),
    'child_epochs': 60,
    "child_batch_size": 64,
}

def get_input_data_generator():
    # preprocessing function executes before rescale
    train_datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0.2,
                                       height_shift_range=0.2, brightness_range=(0.8, 1.2),
                                       shear_range=0.1, zoom_range=0.2,
                                       channel_shift_range=0.2,
                                       fill_mode='reflect', horizontal_flip=True,
                                       vertical_flip=False, rescale=1/255)
    train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE, class_mode='sparse',
                                              color_mode='rgb', batch_size=TRAIN_SET_SIZE, interpolation='lanczos')
    return train


if __name__ == '__main__':
    train = get_input_data_generator()
    x_train, y_train = train.next()
    deepaug = DeepAugment(images=x_train, labels=y_train.reshape(TRAIN_SET_SIZE, 1), config=my_config)
    best_policies = deepaug.optimize(300)
    best_policies.to_csv('best_augment_policies.csv')