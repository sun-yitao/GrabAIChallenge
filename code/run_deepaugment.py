import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from deepaugment.deepaugment import DeepAugment
import sys
sys.path.append(str(Path.cwd() / 'lib'))
from effnet import Effnet

cwd = Path.cwd()
DATA_DIR = cwd.parent / 'data'
TRAIN_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / 'car_data' / 'train'
IMAGE_SIZE = (156, 224)
DATASET_SIZE = 6000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

model = Effnet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), nb_classes=196)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
my_config = {
    'model': model,
    'train_set_size': DATASET_SIZE - 1000,
    'child_epochs': 30,
    'child_batch_size': 256,
    'opt_samples': 1,
    'child_first_train_epochs': 0,
    'pre_aug_weights_path': 'pre_aug_weights.h5',
    'notebook_path': 'notebook.csv',
}

def get_input_data_generator():
    # preprocessing function executes before rescale
    train_datagen = ImageDataGenerator(rescale=1/255)
    train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE, class_mode='sparse', seed=42,
                                              color_mode='rgb', batch_size=DATASET_SIZE, interpolation='lanczos')
    return train


if __name__ == '__main__':
    train = get_input_data_generator()
    x_train, y_train = train.next()
    unique, counts = np.unique(y_train, return_counts=True)
    print(counts)
    y_train = y_train.reshape(DATASET_SIZE, 1)
    deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)
    best_policies = deepaug.optimize(150)
    best_policies.to_csv('best_augment_policies.csv')