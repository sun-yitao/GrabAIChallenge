import os
from pathlib import Path
from glob import glob
import cv2

data_dir = Path.cwd().parent / 'data' / 'downsampled'

def resize_image(input_path):
    im = cv2.imread(input_path)
    ratio = 128 / min(im.shape)
    small_im = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    print(small_im.shape)
    cv2.imwrite(input_path, small_im)

for img in glob(str(data_dir / '**' / '*.jpg'), recursive=True):
    resize_image(img)



