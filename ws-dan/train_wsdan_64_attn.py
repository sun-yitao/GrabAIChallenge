import os
import time
import logging
import warnings
from pathlib import Path
from optparse import OptionParser

import numpy as np
from PIL import Image
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = (512, 512)


def parse_args():
    parser = OptionParser()
    parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                      help='number of data loading workers (default: 16)')
    parser.add_option('--gpu', '--gpu-ids', dest='gpu_ids', default='0',
                      help='IDs of gpu(s) to use in inference, multiple gpus should be seperated with commas')
    parser.add_option('-v', '--verbose', dest='verbose', default=0, type='int',
                      help='show information for each <verbose> iterations (default: 0)')

    parser.add_option('-b', '--batch-size', dest='batch_size', default=32, type='int',
                      help='batch size (default: 32)')
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs (default: 100)')
    parser.add_option('--lr', '--learning-rate', dest='lr', default=0.001, type='float',
                      help='learning rate (default: 0.001)')
    parser.add_option('-m', '--model', dest='model', default='inception',
                      help='model for feature extractor (inception/resnetcbam/efficientnetb3/efficientnetb4')
    
    parser.add_option('-c', '--ckpt', dest='ckpt', default=False,
                      help='path to checkpoint directory if resuming training (default: False)')
    parser.add_option('--dd', '--data-dir', dest='data_dir', default='',
                      help="directory to image folders named 'train' and 'test'")
    parser.add_option('--sd', '--save-dir', dest='save_dir', default=f'./checkpoints/model',
                      help='saving directory of .ckpt models (default: ./checkpoints/model)')
    parser.add_option('--sf', '--save-freq', dest='save_freq', default=1, type='int',
                      help='saving frequency of .ckpt models (default: 1)')

    (options, args) = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_ids
    return options, args


def main():
    options, args = parse_args()
    logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
                        level=logging.INFO)
    warnings.filterwarnings("ignore")

    # Initialize model    
    num_classes = 196
    num_attentions = 64
    start_epoch = 0
    if options.model == 'resnetcbam':
        feature_net = resnet152_cbam(pretrained=True)
    elif options.model == 'efficientnetb3':
        feature_net = EfficientNet.from_pretrained('efficientnet-b3')
    elif options.model == 'efficientnetb4':
        feature_net = EfficientNet.from_name('efficientnet-b4')
    elif options.model == 'inception':
        feature_net = inception_v3(pretrained=True)
    else:
        raise NotImplementedError(f'Invalid model name {options.model}, acceptable values are \
                                    inception/resnetcbam/efficientnetb3/efficientnetb4')
    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions,
                                 net.num_features * net.expansion).to(torch.device(device))

    if options.ckpt:
        ckpt = options.ckpt
        start_epoch = int((ckpt.split('/')[-1]).split('.')[0])

        # Load ckpt and get state_dict
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']

        # Load weights
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(options.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(torch.device(device))
            logging.info('feature_center loaded from {}'.format(options.ckpt))

    # Initialize saving directory
    save_dir = options.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Use cuda
    cudnn.benchmark = True
    net.to(torch.device(device))
    net = nn.DataParallel(net)

    # Load dataset
    cwd = Path.cwd()
    if not options.data_dir:
        data_dir = cwd.parent / 'data' / 'stanford-car-dataset-by-classes-folder' / 'car_data_new_data_in_train_v2'
    else:
        data_dir = options.data_dir

    preprocess_with_augment = transforms.Compose([
        transforms.Resize(size=(image_size[0], image_size[1]), interpolation=Image.LANCZOS),
        #ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    preprocess = transforms.Compose([
        transforms.Resize(size=(image_size[0], image_size[1]), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(str(data_dir / 'train'), transform=preprocess_with_augment)
    validate_dataset = ImageFolder(str(data_dir / 'test'), transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True,
                              num_workers=options.workers, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=options.batch_size * 4, shuffle=False,
                                 num_workers=options.workers, pin_memory=True)

    # Optimizer and loss
    optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, 
                                momentum=0.9, weight_decay=0.00001)
    loss = nn.CrossEntropyLoss()

    # Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                           patience=10, verbose=True, threshold=0.00001)

    # Training
    logging.info('')
    logging.info((f'Start training: Total epochs: {options.epochs}, Batch size: {options.batch_size}, '
                  f'Training size: {len(train_dataset)}, Validation size: {len(validate_dataset)}'))
    best_val_acc = 0
    best_val_epoch = 0
    for epoch in range(start_epoch, options.epochs):
        train(epoch=epoch,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              loss=loss,
              optimizer=optimizer,
              save_freq=options.save_freq,
              save_dir=options.save_dir,
              verbose=options.verbose)
        val_loss, val_acc = validate(data_loader=validate_loader,
                                     net=net,
                                     loss=loss,
                                     verbose=options.verbose)
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
        logging.info(f'Best Validation Accuracy: {best_val_acc}, Epoch: {best_val_epoch + 1}')


def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    feature_center = kwargs['feature_center']
    epoch = kwargs['epoch']
    save_freq = kwargs['save_freq']
    save_dir = kwargs['save_dir']
    verbose = kwargs['verbose']

    # Attention Regularization: LA Loss
    l2_loss = nn.MSELoss()

    # Default Parameters
    beta = 1e-4
    theta_c = 0.5 # crop threshold
    theta_d = 0.5 # drop threshold
    #crop_size = (256, 256)  # size of cropped images for 'See Better'
    crop_size = image_size
    # metrics initialization
    batches = 0
    # Loss on Raw/Crop/Drop Images
    epoch_loss = np.array([0, 0, 0], dtype='float')
    epoch_acc = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]], dtype='float')  # Top-1/3/5 Accuracy for Raw/Crop/Drop Images

    # begin training
    start_time = time.time()
    logging.info('Epoch %03d, Learning Rate %g' %
                 (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()
    for i, (X, y) in enumerate(data_loader):
        batch_start = time.time()

        # obtain data for training
        X = X.to(torch.device(device))
        y = y.to(torch.device(device))

        # Step 1 Original Image
        y_pred, feature_matrix, attention_map = net(X)
        # loss
        batch_loss = loss(y_pred, y) + l2_loss(feature_matrix, feature_center[y])
        epoch_loss[0] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Update Feature Center
        feature_center[y] += beta * (feature_matrix.detach() - feature_center[y])

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[0] += accuracy(y_pred, y, topk=(1, 3, 5))


        # Step 2 Attention Cropping
        with torch.no_grad():
            crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
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
        # crop images forward
        y_pred, _, _ = net(crop_images)

        # loss
        batch_loss = loss(y_pred, y)
        epoch_loss[1] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[1] += accuracy(y_pred, y, topk=(1, 3, 5))


        # Step 3 Attention Dropping
        with torch.no_grad():
            drop_mask = F.upsample_bilinear(
                attention_map, size=(X.size(2), X.size(3))) <= theta_d
            drop_images = X * drop_mask.float()
        # drop images forward
        y_pred, _, _ = net(drop_images)

        # loss
        batch_loss = loss(y_pred, y)
        epoch_loss[2] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[2] += accuracy(y_pred, y, topk=(1, 3, 5))

        # end of this batch
        batches += 1
        batch_end = time.time()
        if verbose and ((i + 1) % verbose == 0):
            logging.info(
                ('\tBatch %d: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f,'
                'Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f') %
                (i + 1,
                 epoch_loss[0] / batches, epoch_acc[0, 0] / batches, 
                 epoch_acc[0, 1] / batches, epoch_acc[0, 2] / batches,
                 epoch_loss[1] / batches, epoch_acc[1, 0] / batches, 
                 epoch_acc[1, 1] / batches, epoch_acc[1, 2] / batches,
                 epoch_loss[2] / batches, epoch_acc[2, 0] / batches, 
                 epoch_acc[2, 1] / batches, epoch_acc[2, 2] / batches,
                 batch_end - batch_start))

    # save checkpoint model
    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'feature_center': feature_center.cpu()},
            os.path.join(save_dir, '%03d.ckpt' % (epoch + 1)))

    # end of this epoch
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info(('Train: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, '
        'Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f') %
        (epoch_loss[0], epoch_acc[0, 0], epoch_acc[0, 1], epoch_acc[0, 2],
         epoch_loss[1], epoch_acc[1, 0], epoch_acc[1, 1], epoch_acc[1, 2],
         epoch_loss[2], epoch_acc[2, 0], epoch_acc[2, 1], epoch_acc[2, 2],
         end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']

    # Default Parameters
    theta_c = 0.5
    crop_size = image_size  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float')  # top - 1, 3, 5

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            batch_start = time.time()
            X = X.to(torch.device(device))
            y = y.to(torch.device(device))

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

            # final prediction
            y_pred = (y_pred_raw + y_pred_crop) / 2

            # loss
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss.item()

            # metrics: top-1, top-3, top-5 error
            epoch_acc += accuracy(y_pred, y, topk=(1, 3, 5))

            # end of this batch
            batches += 1
            batch_end = time.time()
            if verbose and ((i + 1) % verbose == 0):
                logging.info('\tBatch %d: Loss %.5f, Accuracy: Top-1 %.3f, Top-3 %.3f, Top-5 %.3f, Time %3.2f' %
                             (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / batches,
                              epoch_acc[2] / batches, batch_end - batch_start))

    # end of validation
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                 (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))

    return epoch_loss, epoch_acc[0]


if __name__ == '__main__':
    main()
