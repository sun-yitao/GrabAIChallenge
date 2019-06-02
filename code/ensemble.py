import os
from pathlib import Path
from multiprocessing import cpu_count
import random
import argparse

import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import keras
import keras_metrics as km
from keras.layers import Dense, Input
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from lib.se_inception_resnet_v2 import SEInceptionResNetV2
from lib.random_eraser import get_random_eraser
from lib.adabound import AdaBound

IMAGE_SIZE = (363, 525)   # height, width, avg is (483,700) (535,764)
N_CLASSES = 196
BATCH_SIZE = 16

cwd = Path.cwd()
DATA_DIR = cwd.parent / 'data'
TEST_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / \
    'car_data_new_data_in_train_v2' / 'test'
CHECKPOINT_PATH = DATA_DIR / 'checkpoints'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

models = [  # tuple of (model_name, model_path, keras/pytorch)

]


def get_data_generator(model, data_dir):
    if model[2] == 'keras':
        datagen = ImageDataGenerator(rescale=1/255)
        test_gen = datagen.flow_from_directory(data_dir, target_size=IMAGE_SIZE,
                                               color_mode='rgb', batch_size=BATCH_SIZE, interpolation='lanczos')
        return test_gen
    elif model[2] == 'pytorch':
        preprocess = transforms.Compose([
            transforms.Resize(size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(data_dir, transform=preprocess)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=cpu_count(), pin_memory=True)
        return data_loader


def get_prediction_probabilities(model, data_generator):
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_option('--dd', '--dataset_dir', dest='dataset_dir', default=str(TEST_DIR),
                      help='path to test dataset (default: TEST_DIR)')
    args = parser.parse_args()

    for model in models:
        datagen = get_data_generator(model, args.dataset_dir)
        if model[2] == 'keras':
            model = keras.models.load_model(model[1])
        elif model[2] == 'pytorch':

    


if __name__ == '__main__':
    pass
