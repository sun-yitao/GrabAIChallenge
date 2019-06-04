import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = (512, 512)

def main():
    parser = OptionParser()
    parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                      help='number of data loading workers (default: 16)')
    parser.add_option('-m', '--model', dest='model', default='inception',
                      help='CNN model')
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs (default: 100)')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=32, type='int',
                      help='batch size (default: 32)')
    parser.add_option('-c', '--ckpt', dest='ckpt', default=False,
                      help='load checkpoint model (default: False)')
    parser.add_option('-v', '--verbose', dest='verbose', default=0, type='int',
                      help='show information for each <verbose> iterations (default: 0)')

    parser.add_option('--lr', '--learning-rate', dest='lr', default=0.001, type='float',
                      help='learning rate (default: 1e-3)')
    parser.add_option('--sf', '--save-freq', dest='save_freq', default=1, type='int',
                      help='saving frequency of .ckpt models (default: 1)')
    parser.add_option('--sd', '--save-dir', dest='save_dir', default=f'./checkpoints/model',
                      help='saving directory of .ckpt models (default: ./models)')
    parser.add_option('--init', '--initial-training', dest='initial_training', default=False,
                      help='train from beginning or resume training (default: False)')
    (options, args) = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")
    

    num_classes = 196
    start_epoch = 0
    if options.model == 'resnet':
        net = resnet101(pretrained=True)
    elif options.model == 'efficientnet':
        net = EfficientNet.from_pretrained('efficientnet-b3')
        
    elif options.model == 'inception':
        net = inception_v3(pretrained=True)
    else:
        raise NotImplementedError('Invalid model name, choose from resnet, inception or efficientnet')
    net._fc = nn.Linear(net._fc.in_features, num_classes)
    if options.ckpt:
        ckpt = options.ckpt

        if options.initial_training:
            # Get Name (epoch)
            epoch_name = (ckpt.split('/')[-1]).split('.')[0]
            start_epoch = int(epoch_name)

        # Load ckpt and get state_dict
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']

        # Load weights
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(options.ckpt))

    ##################################
    # Initialize saving directory
    ##################################
    save_dir = options.save_dir
    os.makedirs(save_dir, exist_ok=True)

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.to(torch.device(device))
    net = nn.DataParallel(net)

    ##################################
    # Load dataset
    ##################################
    preprocess = transforms.Compose([
        transforms.Resize(size=(image_size[0], image_size[1]), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    cwd = Path.cwd()
    data_dir = cwd.parent / 'GrabAIChallenge' / 'data' / \
        'stanford-car-dataset-by-classes-folder' / 'car_data_new_data_in_train_v2'

    train_dataset = ImageFolder(str(data_dir / 'train'), transform=preprocess)
    validate_dataset = ImageFolder(str(data_dir / 'test'), transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True,
                              num_workers=options.workers, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=options.batch_size, shuffle=False,
                                 num_workers=options.workers, pin_memory=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, 
                                momentum=0.9, weight_decay=0.00001)
    loss = nn.CrossEntropyLoss() #TODO add class weight

    ##################################
    # Learning rate scheduling
    ##################################
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                           patience=10, verbose=True, threshold=0.00001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # TRAINING
    ##################################
    logging.info('')
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(options.epochs, options.batch_size, len(train_dataset), len(validate_dataset)))

    for epoch in range(start_epoch, options.epochs):
        train(epoch=epoch,
              data_loader=train_loader,
              net=net,
              loss=loss,
              optimizer=optimizer,
              save_freq=options.save_freq,
              save_dir=options.save_dir,
              verbose=options.verbose)
        val_loss = validate(data_loader=validate_loader,
                            net=net,
                            loss=loss,
                            verbose=options.verbose)
        scheduler.step(val_loss)


def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    epoch = kwargs['epoch']
    save_freq = kwargs['save_freq']
    save_dir = kwargs['save_dir']
    verbose = kwargs['verbose']

    # begin training
    start_time = time.time()
    epoch_loss = 0
    epoch_acc = 0
    logging.info('Epoch %03d, Learning Rate %g' %
                 (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()

    for i, (X, y) in enumerate(data_loader):
        batch_start = time.time()

        # obtain data for training
        X = X.to(torch.device(device))
        y = y.to(torch.device(device))

        ##################################
        # Raw Image
        ##################################
        y_pred = net(X)
        # loss
        batch_loss = loss(y_pred, y)
        epoch_loss += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_acc[0] += accuracy(y_pred, y, topk=(1, 3, 5))

        # end of this batch
        batches += 1
        batch_end = time.time()
        if verbose and ((i + 1) % verbose == 0):
            logging.info(('\tBatch %d: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f,'
                'Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f') %
                         (i + 1,
                          epoch_loss[0] / batches, epoch_acc[0, 0] /
                          batches, epoch_acc[0, 1] /
                          batches, epoch_acc[0, 2] / batches,
                          epoch_loss[1] / batches, epoch_acc[1, 0] /
                          batches, epoch_acc[1, 1] /
                          batches, epoch_acc[1, 2] / batches,
                          epoch_loss[2] / batches, epoch_acc[2, 0] /
                          batches, epoch_acc[2, 1] /
                          batches, epoch_acc[2, 2] / batches,
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
                  epoch_loss[1], epoch_acc[1,
                                           0], epoch_acc[1, 1], epoch_acc[1, 2],
                  epoch_loss[2], epoch_acc[2,
                                           0], epoch_acc[2, 1], epoch_acc[2, 2],
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

            # obtain data
            X = X.to(torch.device(device))
            y = y.to(torch.device(device))

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, feature_matrix, attention_map = net(X)
            ##################################
            # Object Localization and Refinement
            ##################################
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
                             (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / \
                                  batches, epoch_acc[2] / batches, batch_end - batch_start))

    # end of validation
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                 (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))
    logging.info('')

    return epoch_loss


if __name__ == '__main__':
    main()
