import os
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import keras
import keras_metrics as km
from keras.layers import Dense, Input
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from se_inception_resnet_v2 import SEInceptionResNetV2
from random_eraser import get_random_eraser
from adabound import AdaBound


MODEL_NAME = 'SEResNext'
EPOCHS = 200  # only for calculation of lr decay
IMAGE_SIZE = (483, 700)  # height, width
N_CLASSES = 196
LR_START = 0.01
BATCH_SIZE = 32

cwd = Path.cwd()
DATA_DIR = cwd.parent / 'data'
TRAIN_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / 'car_data' / 'train'
TEST_DIR = DATA_DIR / 'stanford-car-dataset-by-classes-folder' / 'car_data' / 'test'
CHECKPOINT_PATH = DATA_DIR / 'checkpoints' / 'baseline_cnn' / MODEL_NAME
LOG_DIR = DATA_DIR / 'logs' / 'baseline_cnn' / MODEL_NAME


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


def get_input_data_generators():
    # preprocessing function executes before rescale
    train_datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0.2,
                                       height_shift_range=0.2, brightness_range=(0.8, 1.2),
                                       shear_range=0.1, zoom_range=0.2,
                                       channel_shift_range=0.2,
                                       fill_mode='reflect', horizontal_flip=True,
                                       vertical_flip=False, rescale=1/255,
                                       preprocessing_function=get_random_eraser(p=0.8, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                                                                                v_l=0, v_h=255, pixel_level=True))
    test_datagen = ImageDataGenerator(rescale=1/255)
    train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE,
                                              color_mode='rgb', batch_size=BATCH_SIZE, interpolation='lanczos')
    test = test_datagen.flow_from_directory(TEST_DIR, target_size=IMAGE_SIZE,
                                            color_mode='rgb', batch_size=BATCH_SIZE, interpolation='lanczos')
    return train, test


def get_model():
    input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model = Xception(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                          include_top=False,
                          weights=None,
                          input_tensor=input_tensor,
                          pooling='avg',
                          classes=N_CLASSES)
    x = base_model.output
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    decay = LR_START / EPOCHS
    optm = AdaBound(lr=0.001,
                    final_lr=0.01,
                    gamma=1e-03,
                    weight_decay=decay,
                    amsbound=False)
    precision = km.categorical_precision()
    recall = km.categorical_recall()
    f1_score = km.categorical_f1_score()
    model.compile(optimizer=optm,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall, f1_score])
    return model


def get_callbacks():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_PATH, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                                                  verbose=1, mode='auto',  # min_delta=0.001,
                                                  cooldown=0, min_lr=0)
    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(LOG_DIR)
    return [ckpt, reduce_lr, tensorboard]


if __name__ == '__main__':
    train, test = get_input_data_generators()
    model = get_model()
    callbacks = get_callbacks()
    class_weights = compute_class_weight(
        'balanced', np.arange(0, N_CLASSES), train.classes)
    model.fit_generator(train, steps_per_epoch=len(train), epochs=1000,
                        validation_data=test, validation_steps=len(test),
                        callbacks=callbacks)
