import os
from pathlib import Path
from glob import glob
from optparse import OptionParser

from tqdm import tqdm
from PIL import Image
import numpy as np
import keras
from keras.preprocessing.image import array_to_img, img_to_array

cwd = Path.cwd()
image_dir = cwd.parent / 'data' / 'super_resolution'
model_path = cwd.parent / 'data' / 'x4_DBPN.h5'

def parse_args():
    parser = OptionParser()
    parser.add_option('--gpu', '--gpu-ids', dest='gpu_ids', default='0',
                      help='IDs of gpu(s) to use in inference, multiple gpus should be seperated with commas')
    parser.add_option('--dd', '--data-dir', dest='data_dir', default=str(image_dir),
                      help='directory to images to run super resolution')
    parser.add_option('--mp', '--model-path', dest='model_path', default=str(model_path),
                      help='Path to saved model checkpoint (default: ./checkpoints/model.pth)')
    (options, args) = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_ids
    return options, args


if __name__ == '__main__':
    max_img_area_to_run_superres = 128*256
    options, args = parse_args()
    images = glob(str(Path(options.data_dir) / '**' / '*.jpg'), recursive=True)
    model = keras.models.load_model(options.model_path)
    images_to_superres = []
    for image in tqdm(images):
        im = Image.open(image)
        original_im_area = im.size[0] * im.size[1]
        if original_im_area < max_img_area_to_run_superres:
            np_image = img_to_array(im)
            np_image /= 255
            np_image = np.expand_dims(np_image, axis=0)
            out = model.predict(np_image)
            out = np.clip(out, 0, 1)
            out *= 255
            im = array_to_img(out[0])
            im.save(str(image))

