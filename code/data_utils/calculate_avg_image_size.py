from pathlib import Path
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm


cwd = Path.cwd()
data_dir = cwd.parent.parent / 'data' / 'stanford-car-dataset-by-classes-folder' / 'car_data_new_data_in_train'
img_paths = glob(str(data_dir / '**' / '*.jpg'), recursive=True)
img_sizes = np.empty((len(img_paths),2))
for n, img_path in enumerate(tqdm(img_paths)):
    im = Image.open(img_path)
    img_sizes[n] = np.array(im.size)

print('Average Image Size: ', np.mean(img_sizes, axis=0))
    
