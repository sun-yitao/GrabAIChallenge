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

MODEL_PATH = ''
IMAGE_SIZE = (363, 525)  # height, width


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
    test = get_input_data_generators_from_directory('', batch_size=32)
    model.predict_generator
