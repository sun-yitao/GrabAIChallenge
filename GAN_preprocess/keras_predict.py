import os
import pathlib
from PIL import Image
import numpy as np
import keras
from keras.preprocessing.image import array_to_img, img_to_array
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cwd = pathlib.Path.cwd()
image_dir = cwd / 'Input' / 'test'
images = image_dir.glob('*.jpg')
model_path = cwd / 'x4_DBPN.h5'
model = keras.models.load_model(str(model_path))
for image in images:
    im = Image.open(image)
    np_image = img_to_array(im)
    np_image /= 255
    np_image = np.expand_dims(np_image, axis=0)
    out = model.predict(np_image)
    out = np.clip(out, 0, 1)
    out *= 255
    im = array_to_img(out[0])
    im.save(str(cwd / 'Results' / 'test' / image.name))

