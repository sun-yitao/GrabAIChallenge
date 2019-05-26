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



## Results

| Model Name                                                    | Training Accuracy | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 Score |
| ------------------------------------------------------------- | ----------------- | ------------------- | -------------------- | ----------------- | ------------------- |
| CNN Baseline with Xception, random cutout, adabound optimiser | 0.9811            | 0.9366              | 1.0000               | 0.9091            | 0.9524              |
| Weakly Supervised Data Augmentation Network                   | 99.76             | 91.28               |                      |                   |                     |
|                                                               |                   |                     |                      |                   |                     |
|                                                               |                   |                     |                      |                   |                     |
|                                                               |                   |                     |                      |                   |                     |
