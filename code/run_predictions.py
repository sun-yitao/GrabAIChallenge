import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_metrics as km
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from lib.adabound import AdaBound

MODEL_PATH = '/Users/sunyitao/Documents/Projects/ML_Projects/GrabAIChallenge/data/checkpoints/baseline_cnn/Xception_Imagenet/model.60-0.94.h5'
IMAGE_SIZE = (363, 525)  # height, width

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


def get_input_data_generators_from_directory(test_directory, batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1/255)
    test = test_datagen.flow_from_directory(test_directory, target_size=IMAGE_SIZE, shuffle=False,
                                            color_mode='rgb', batch_size=batch_size, interpolation='lanczos')
    return test


def get_input_data_generator_from_csv(csv_path, test_directory=None, x_col='filename', y_col='class', batch_size=32):
    """Reads csv of image filenames and labels

    Args:
        csv_path (str): path to csv file
        test_directory (str or None): path to test directory, if None, image filenames in csv needs to be full path
        x_col and y_col (str): name of csv header for image filename and label
    """
    df = pd.read_csv(csv_path)
    test_datagen = ImageDataGenerator(rescale=1/255)
    test = test_datagen.flow_from_dataframe(df, directory=test_directory, x_col=x_col, y_col=y_col,
                                            target_size=IMAGE_SIZE, shuffle=False, drop_duplicates=False,
                                            color_mode='rgb', batch_size=batch_size, interpolation='lanczos')
    return test


if __name__ == '__main__':
    model = keras.models.load_model(
        MODEL_PATH, custom_objects={'AdaBound': AdaBound})
    test = get_input_data_generators_from_directory('/Users/sunyitao/Documents/Projects/ML_Projects/GrabAIChallenge/data/stanford-car-dataset-by-classes-folder/car_data/new_data_cleaned', batch_size=32)
    model.evaluate_generator(test, steps=len(test), verbose=1)
