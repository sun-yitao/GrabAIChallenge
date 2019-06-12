import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import logging
import warnings
from pathlib import Path
from optparse import OptionParser
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
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
from dataset import CustomDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = (512, 512)

#TODO
"""
1. extract probs given image directory
2. save in csv image_path, pred_raw, pred_crop, (ground truth)
3. run xgboost BayesCV training in separate script
4. predict/ evaluate using extract probs -> xgb
"""

def parse_args():
    parser = OptionParser()
    parser.add_option('-j', '--workers', dest='workers', default=cpu_count(), type='int',
                      help='number of data loading workers (default: n_cpus)')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=32, type='int',
                      help='batch size (default: 32)')
    parser.add_option('-de', '--do-eval', dest='do_eval', default=True,
                      help='Whether to evaluate single model performance if images are organised \
                           in class folders inside data_dir. If False, data_dir contains images directly')
    parser.add_option('--dd', '--data-dir', dest='data_dir', default='',
                      help="directory to images or class folders containing images")
    parser.add_option('--cd', '--ckpt-dir', dest='ckpt_dir', default=f'./checkpoints',
                      help='saving directory of .ckpt models (default: ./checkpoints)')
    parser.add_option('--od', '--output-dir', dest='output_dir', default=f'./output',
                      help='saving directory of extracted class probabilities csv file')
    (options, args) = parser.parse_args()
    return options, args


def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")
    options, args = parse_args()
    feature_nets = {
        'efficientnet': os.path.join(options.save_dir, 'efficientnet.pth'),
        'resnet152cbam': os.path.join(options.save_dir, 'resnet152cbam.pth'),
        'inceptionv3': os.path.join(options.save_dir, 'inceptionv3.pth'),
    }



    logging.info(f'Extract Probabilities: Batch size: {options.batch_size}, Dataset size: {len(dataset)}')
    for feature_net_name, ckpt_path in feature_nets.items():
        extract_class_probabilities(options, feature_net_name, ckpt_path)
        if device == 'cuda':
            torch.cuda.empty_cache()


def extract_class_probabilities(options, feature_net_name, ckpt_path):
    """
    Given feature net name and ckpt path, predicts probabilties
    """
    # Initialize model    
    num_classes = 196
    num_attentions = 32
    
    if feature_net_name == 'resnet':
        feature_net = resnet152_cbam(pretrained=True)
    elif feature_net_name == 'efficientnet':
        feature_net = EfficientNet.from_pretrained('efficientnet-b3')
    elif feature_net_name == 'inception':
        feature_net = inception_v3(pretrained=True)
    else:
        raise RuntimeError('Invalid model name')

    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt_path))
    # load feature center
    feature_center = checkpoint['feature_center'].to(torch.device(device))
    logging.info('feature_center loaded from {}'.format(options.ckpt))

    # Use cuda
    cudnn.benchmark = True
    net.to(torch.device(device))
    net = nn.DataParallel(net)

    # Load dataset
    preprocess = transforms.Compose([
        transforms.Resize(size=(image_size[0], image_size[1]), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    if options.do_eval:
        dataset = ImageFolder(str(options.data_dir), transform=preprocess)
        image_list = [sample[0] for sample in dataset.samples]
    else:
        # returns image without label
        dataset = CustomDataset(str(options.data_dir), shape=image_size)
        image_list = dataset.image_list
    data_loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False,
                                num_workers=options.workers, pin_memory=True)

    logging.info(f'Extract Probabilities: Batch size: {options.batch_size}, Dataset size: {len(dataset)}')
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
    y_pred_raw_numpy = np.zeros((len(dataset), 196))
    y_pred_crop_numpy = np.zeros((len(dataset), 196))

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
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
            y_pred_raw_numpy[i*options.batch_size:(i+1)*options.batch_size] = y_pred_raw.numpy()
            y_pred_crop_numpy[i*options.batch_size:(i+1)*options.batch_size] = y_pred_crop.numpy()

            if options.do_eval:
                # loss
                batch_loss = loss(y_pred, y)
                epoch_loss += batch_loss.item()

                # metrics: top-1, top-3, top-5 error
                epoch_acc += accuracy(y_pred, y, topk=(1, 3, 5))

    end_time = time.time()
    output_csv_path = os.path.join(options.output_dir, feature_net_name + '_probs.csv')
    if options.do_eval:
        logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                    (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))
        to_csv(image_list, y_pred_raw_numpy, y_pred_crop_numpy, output_csv_path
              [sample[1] for sample in dataset.samples])
    else:
        to_csv(image_list, y_pred_raw_numpy, y_pred_crop_numpy, output_csv_path)



def to_csv(image_list, pred_raw, pred_crop, csv_path, ground_truth=None):
    logging.info('Saving class probabilities to csv')
    if ground_truth:
        df = pd.DataFrame({'image_path': image_list,
                            'pred_raw': pred_raw,
                            'pred_crop': pred_crop,
                            'label': ground_truth})
    else:
        df = pd.DataFrame({'image_path': image_list,
                            'pred_raw': pred_raw,
                            'pred_crop': pred_crop})
    df.to_csv(csv_path)


if __name__ == '__main__':
    main()
