import os
import time
import logging
import warnings
from pathlib import Path
from optparse import OptionParser
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy.special import softmax
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet

from autoaugment import ImageNetPolicy
from utils import accuracy
from models import *
from dataset import UnlabelledDataset, CsvDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = (512, 512)


def parse_args():
    parser = OptionParser()
    parser.add_option('-j', '--workers', dest='workers', default=cpu_count(), type='int',
                      help='number of data loading workers (default: n_cpus)')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=32, type='int',
                      help='batch size (default: 32)')
    parser.add_option('--fn', '--feature-net', dest='feature_net_name', default='efficientnetb3',
                      help='Name of base model. Accepted values are inception/resnet152cbam/efficientnetb3 (default: efficientnetb3)')
    parser.add_option('--gpu', '--gpu-ids', dest='gpu_ids', default='0',
                      help='IDs of gpu(s) to use in inference, multiple gpus should be seperated with commas')
    

    parser.add_option('--de', '--do-eval', dest='do_eval', default=True,
                      help='If labels are provided, set True to evaluate metrics (default: True)')
    parser.add_option('--csv', '--csv-labels-path', dest='csv_labels_path', default='folder',
                      help='If eval mode is set, set to "folder" to read labels from folders \
                            with classnames. Set to csv path to read labels from csv (default: folder)')
    parser.add_option('--csv-headings', dest='csv_headings', default='image,label',
                      help='heading of image filepath and label column in csv')     

    parser.add_option('--dd', '--data-dir', dest='data_dir', default='data',
                      help='directory to images to run evaluation/ prediction')
    parser.add_option('--cp', '--ckpt-path', dest='ckpt_path', default='./checkpoints/model.pth',
                      help='Path to saved model checkpoint (default: ./checkpoints/model.pth)')
    parser.add_option('--od', '--output-dir', dest='output_dir', default='./output',
                      help='saving directory of extracted class probabilities csv file')
    (options, args) = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_ids
    return options, args


def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")
    options, args = parse_args()
    predict_class_probabilities(options)


def prepare_dataloader(options):
    """Loads data from folder containing labelled folders of images/ csv containing filepaths and labels 0-195/
    Unlabelled folder of images
    """
    preprocess = transforms.Compose([
        transforms.Resize(size=(image_size[0], image_size[1]), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    if options.do_eval:
        if options.csv_labels_path == 'folder':
            dataset = ImageFolder(str(options.data_dir), transform=preprocess)
        else:
            dataset = CsvDataset(str(options.data_dir), options.csv_labels_path, 
                                 options.csv_headings, transform=preprocess)
        image_list = [sample[0] for sample in dataset.samples]

    else:
        # returns image without label
        dataset = UnlabelledDataset(str(options.data_dir), transform=preprocess, shape=image_size)
        image_list = dataset.image_list
    data_loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False,
                             num_workers=options.workers, pin_memory=True)
    logging.info(f'Extract Probabilities: Batch size: {options.batch_size}, Dataset size: {len(dataset)}')
    return dataset, data_loader, image_list


def predict_class_probabilities(options):
    """Predicts class probabilities and optionally evaluates accuracy, precision, 
    recall and f1 score if labels are provided

    Args:
        options: parsed arguments

    Returns:

    """
    # Initialize model    
    num_classes = 196
    num_attentions = 64

    if options.feature_net_name == 'resnet152cbam':
        feature_net = resnet152_cbam(pretrained=True)
    elif options.feature_net_name == 'efficientnetb3':
        feature_net = EfficientNet.from_pretrained('efficientnet-b3')
    elif options.feature_net_name == 'inception':
        feature_net = inception_v3(pretrained=True)
    else:
        raise RuntimeError('Invalid model name')

    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

    # Load ckpt and get state_dict
    checkpoint = torch.load(options.ckpt_path)
    state_dict = checkpoint['state_dict']
    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(options.ckpt_path))
    # load feature center
    feature_center = checkpoint['feature_center'].to(torch.device(device))
    logging.info('feature_center loaded from {}'.format(options.ckpt_path))

    # Use cuda
    cudnn.benchmark = True
    net.to(torch.device(device))
    net = nn.DataParallel(net)

    # Load dataset
    dataset, data_loader, image_list = prepare_dataloader(options)
    # Default Parameters
    theta_c = 0.5
    crop_size = image_size  # size of cropped images for 'See Better'
    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float')  # top - 1, 3, 5
    loss = nn.CrossEntropyLoss()

    start_time = time.time()
    net.eval()
    y_pred_average = np.zeros((len(dataset), 196))

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            if options.do_eval:
                X, y = sample
                y = y.to(torch.device(device))
            else:
                X = sample
            X = X.to(torch.device(device))
            
            # Raw Image
            y_pred_raw, feature_matrix, attention_map = net(X)
            # Object Localization and Refinement
            crop_mask = F.upsample_bilinear(
                attention_map, size=(X.size(2), X.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
                crop_images.append(F.upsample_bilinear(
                    X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
            crop_images = torch.cat(crop_images, dim=0)
            y_pred_crop, _, _ = net(crop_images)

            y_pred = (y_pred_raw + y_pred_crop) / 2
            y_pred_average[i*options.batch_size:(i+1)*options.batch_size] = y_pred.cpu().numpy()
            batches += 1
            if options.do_eval:
                # loss
                batch_loss = loss(y_pred, y)
                epoch_loss += batch_loss.item()

                # metrics: top-1, top-3, top-5 error
                epoch_acc += accuracy(y_pred, y, topk=(1, 3, 5))

    end_time = time.time()
    if options.do_eval:
        epoch_loss /= batches
        epoch_acc /= batches
        logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.4f, Top-3 %.4f, Top-5 %.4f, Time %3.2f' %
                    (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))
        ground_truth = [sample[1] for sample in dataset.samples]
        precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, np.argmax(y_pred_average, axis=1), average='micro')
        logging.info(f'Precision: {precision}, Recall: {recall}, Micro F1: {f1}')
        y_pred_average = softmax(y_pred_average, axis=1)
        save_predictions(image_list, y_pred_average, options, ground_truth=ground_truth)
    else:
        save_predictions(image_list, y_pred_average, options)



def save_predictions(image_list, predicted_probabilities, options, ground_truth=None):
    logging.info('Saving class probabilities to csv')
    if ground_truth:
        df = pd.DataFrame({'image_path': image_list,
                            'label': ground_truth})
    else:
        df = pd.DataFrame({'image_path': image_list})
    os.makedirs(options.output_dir, exist_ok=True)
    df.to_csv(os.path.join(options.output_dir, options.feature_net_name + '_ImageList.csv'))
    np.save(os.path.join(options.output_dir, options.feature_net_name + '_PredictedProbabilites.npy'),
            predicted_probabilities)


if __name__ == '__main__':
    main()
