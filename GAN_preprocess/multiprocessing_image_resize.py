import os
from pathlib import Path
import cv2

data_dir = Path.cwd().parent / 'data' / 'downsampled'

def resize_image(input_path):
    im = cv2.imread(input_path)
    ratio = 128 / min(im.shape)
    small_im = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    cv2.imwrite(input_path, small_im)

for img in data_dir.glob('*.jpg'):
    resize_image(str(img))



