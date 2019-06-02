# GrabAIChallenge

AI For SEA Challenge

## Prepare Dataset

First download the stanford cars dataset by classes folder at [https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)

Unzip and put into a directory named 'data'

We will combine the provided train and test dataset and do a 75/25 train test split using rebuild_dataset.py

Train size: 12208

Test size: 3997

Duplicates have been removed using Gemini software.

We added new data from google images using the script code/g_images_download.py

Some of these new images do not contain relevant data, we cleaned around 25 image folders manually and fine-tuned a pretrained Xception to classify these new images as wanted or unwanted. We used this model to help us clean the rest of the data using code/predict_unwanted_images.py

The new data is also cleaned manually as there are some wrongly classified images eg: convertible vs coupe, Dodge Challenger vs Dodge Charger SRT.

## Augmentation

Gaussian noise cutout

deepaugment



GAN day to night:

dataset: http://www.carlib.net/?page_id=35, nuscenes

## Model

Xception

Weakly Supervised Data Augmentation Network

## Results

#### CNN Baseline

| Model Name                                                | Training Accuracy | Validation Accuracy |
| --------------------------------------------------------- | ----------------- | ------------------- |
| CNN Baseline: Xception, random cutout, adabound optimiser | 0.9811            | 0.9366              |
| CNN Baseline New Data Classweights                        | 0.9538            | 0.92265             |
| CNN Baseline New Data V2 Classweights                     | 0.9725            | 0.93507             |
| CNN Baseline New Data V2 Classweights New Aspect Ratio    | 0.9836            | 0.93305             |
|                                                           |                   |                     |



#### Weakly Supervised Data Augmentation Network

| Model Name                                                          | Training Accuracy | Validation Accuracy |
| ------------------------------------------------------------------- | ----------------- | ------------------- |
| Original Implementation                                             | 99.76             | 91.28               |
| Change crop size to original image size                             | 99.76             | 93.29               |
| Change LR schedule and remove double input preprocessing (epoch 31) | 99.75             | 94.78               |
|                                                                     |                   |                     |
|                                                                     |                   |                     |
