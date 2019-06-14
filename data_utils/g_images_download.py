import os
from pathlib import Path
from google_images_download import google_images_download


cwd = Path.cwd()
data_dir = cwd.parent / 'data' / 'stanford-car-dataset-by-classes-folder' / 'car_data'
output_dir = data_dir / 'new_data'
os.makedirs(str(output_dir), exist_ok=True)
response = google_images_download.googleimagesdownload()

for folder in (data_dir / 'train').iterdir():
    if folder.is_dir():
        class_name = folder.name
        arguments = {'keywords': str(class_name), 'limit': 100, 'print_paths': True, 
                     'output_directory': str(data_dir / 'new_data')}
        paths = response.download(arguments)