import os
from pathlib import Path
from shutil import move
from multiprocessing import cpu_count

import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import keras
from keras.layers import Dense, Input
from keras.applications.xception import Xception
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

from lib.random_eraser import get_random_eraser
from lib.adabound import AdaBound

"""
Iterates through new dataset from google images and predicts if images are wanted or unwanted
wanted images show the exterior of the car and unwanted images are everything else
if predicted as unwanted, move image into unwanted folder
"""

IMAGE_SIZE = (363, 525)  # height, width, avg is 483, 700
BATCH_SIZE = 16

cwd = Path.cwd()
DATA_DIR = cwd.parent / 'data'
TRAIN_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / \
    'car_data' / 'new_data_2'


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


if __name__ == '__main__':
    model = keras.models.load_model(str(DATA_DIR / 'checkpoints' / 'data_cleaning_model' / 'model.01-1.00.h5'))
    for folder in tqdm(list(TRAIN_DIR.iterdir())):
        if folder.is_dir():
            images = list(folder.glob('*.jpg'))
            img_array = np.zeros((len(images), IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            for n, image in enumerate(images):
                try:
                    pil_im = load_img(str(image), target_size=IMAGE_SIZE, interpolation='lanczos')
                except OSError:
                    continue
                img_array[n] = img_to_array(pil_im)
            img_array = img_array / 255
            preds = np.argmax(model.predict(img_array), axis=1)
            unwanted_image_indices = np.argwhere(preds == 0) # unwanted class
            os.makedirs(str(folder / 'unwanted'), exist_ok=True)
            for idx in unwanted_image_indices:
                move(str(images[idx[0]]), str(folder / 'unwanted'))