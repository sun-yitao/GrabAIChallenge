import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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


def plot_confusion_matrix(y_true, y_pred, categories,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,
                          number_categories=False,
                          normalize_axis=1):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize and normalize_axis == 1:
        cm = cm.astype('float') / cm.sum(axis=normalize_axis)[:, np.newaxis]
    elif normalize and normalize_axis == 0:
        cm = cm.astype('float') / cm.sum(axis=normalize_axis)[np.newaxis, :]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=categories, yticklabels=categories,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.set_size_inches(16,16)
    return ax


if __name__ == '__main__':
    precision = km.categorical_precision()
    recall = km.categorical_recall()
    f1_score = km.categorical_f1_score()
    model = keras.models.load_model(
        MODEL_PATH, custom_objects={'AdaBound': AdaBound, 'categorical_precision': precision, 'categorical_recall': recall, 'categorical_f1_score': f1_score})
    test = get_input_data_generators_from_directory(
        '/Users/sunyitao/Documents/Projects/ML_Projects/GrabAIChallenge/data/stanford-car-dataset-by-classes-folder/car_data/new_data_cleaned', batch_size=32)
    results = model.evaluate_generator(test, steps=len(test), verbose=1)
    print(results)
    for metric, score in zip(model.metric_names, results):
        print(metric, score)
