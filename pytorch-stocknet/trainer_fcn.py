import argparse
import logging
import os
import os.path as osp
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
# from torchsummary import summary

# from unet import UNet
from model import Stocknet

from torch.utils.tensorboard import SummaryWriter
from dataset import StocknetDataset
# from utils.dataset_with_std import BasicDataset
from torch.utils.data import DataLoader

from copy import deepcopy

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

"""
Anish's edit
------------
needed for subsetting datasets.
"""
from torch.utils.data import Subset

# set up paths.
dir_img = configs["dir_img"]
dir_mask = configs["dir_mask"]
dir_checkpoint = configs["dir_checkpoint"]

best_model_state = None
# optimizer_state = None
# scheduler_state = None
current_epoch = 0 # Number of epochs completed.
current_loss = np.inf
checkpoint = dict()
current_best_val_score = 0 # this is dice score for now. 0 is worst.
epoch_of_best_model = 0
best_saved = True

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              last="last.tar",
              best="best.tar",
              log_dir="runs/FCN",
              ):

    global best_model_state, \
         current_epoch, current_loss, current_best_val_score, epoch_of_best_model, \
             best_saved
        # optimizer_state, scheduler_state,\

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    """
    Anish's edit
    ------------
    we don't want random test and train. Instead, we are taking initial
    3070 images for train, last 1030 for test, and remaining 911 for
    validation. This is done so that training, validation, and testing
    are done on mutually exclusive patient sets.
    """
    n_val = 911
    n_train = 3070
    train_indices = [i for i in range(n_train)]
    val_indices = [i for i in range(n_train, n_train + n_val)]
    train, val = Subset(dataset, train_indices), Subset(dataset, val_indices)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # writer = SummaryWriter(log_dir=log_dir, comment=f'_{net.name}_LR_{lr}_BS_SEPOCH_{current_epoch}_EEPOCH_{epochs}')
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    logging.info(f'''Starting training:
        Model:           {net.name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Freeze:          {net.freeze}
    ''')

    # optimizer = optim.SGD(net.parameters(), lr=lr)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # restore state if needed
    if "optimizer" in checkpoint:
        checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = lr
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # if "scheduler" in checkpoint:
    #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # for optimzer, and scheduler we only need a reference since we need the last values.
    # scheduler_state = scheduler.state_dict()
    # optimizer_state = optimizer.state_dict()
    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # start or resume training
    for epoch in range(current_epoch, epochs):
        net.train()

        epoch_loss = 0
        """
        Anish's edit
        ------------
        in tqdm, set leave=True instead of False, and position=0. This will prevent
        each update being made in new line in google colab.
        """
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', position=0, leave=True) as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    # Validation

                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score = eval_net(net, val_loader, device)
                    # scheduler.step(val_score)

                    # update best model if needed.
                    if val_score > current_best_val_score:
                        # new best found.
                        logging.info(f"""
                        New Best Model!
                        Previous Dice Index = {current_best_val_score}
                        New Dice Index = {val_score}
                        Improvement = {val_score - current_best_val_score}
                        """)
                        best_saved = False
                        current_best_val_score = val_score
                        best_model_state = deepcopy(net.state_dict())
                        epoch_of_best_model = epoch + 1


                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        current_epoch = epoch + 1
        # save only if we have an unsaved best.
        if save_cp:
            try:
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
        
            torch.save({
            'current_epoch': current_epoch,
            'model_state_dict': net.state_dict(), # model state after the last completed epoch. can be used to resume.
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            'current_loss': current_loss,
            'current_best_val_score': current_best_val_score,
            "epoch_of_best_model": epoch_of_best_model,
            }, osp.join(dir_checkpoint, last))

            if not best_saved:
                torch.save({
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'current_best_val_score': current_best_val_score,
                "epoch_of_best_model": epoch_of_best_model,
                }, osp.join(dir_checkpoint, best))
                
                logging.info(f'Saved new best model with val Dice Score {current_best_val_score}.')
                best_saved = True

    # print statuses
    logging.info(f"""
    Training done.
    Best validation Dice Score: {current_best_val_score}
    Best model obtained in epoch {epoch_of_best_model}
    Model has been trained for {current_epoch} epochs in total.
    
    Wrapping up...
    """)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the FCNs on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", type=str, default="fcn32", help="One of fcn32, fcn16, fcn8, fcn4",
                        dest="chosen_model")
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-bc', '--best-checkpoint', dest='best', type=str, default="best.tar",
                        help="Name of file to store best model in.")
    parser.add_argument('-lc', '--last-checkpoint', type=str, dest='last', default="last.tar",
                        help='Name of file to store last model, and states in.')
    parser.add_argument('-cc', '--checkpoint-dir', type=str, dest="cc", default="checkpoints",
                        help="Checkpoint root directory path")
    parser.add_argument("-fz", '--freeze-vgg', type=int, default=0, dest="freeze",
                        help="whether to freeze parent model's parameters (0 / 1)")
    parser.add_argument("-lg", "--log-dir", dest="log_dir", type=str, default="runs/FCN",
                        help="directory for logging tensorboard data")
    parser.add_argument("-ns", "--not-strict", dest="strict", action="store_false",
                        help="use this flag if params are to be loaded from saved ones nonstrict")

    return parser.parse_args()


if __name__ == '__main__':
    # global dir_checkpoint
    # global best_model_state, checkpoint, current_epoch, current_loss, current_best_val_score, epoch_of_best_model

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    available_models = {
        "fcn32": FCN32s,
        "fcn16": FCN16s,
        "fcn8": FCN8s,
        "fcn4": FCN4s,
    }
    try:
        chosen_model = available_models[args.chosen_model]
    except:
        print("Unknown model!")
        raise

    dir_checkpoint = osp.join(args.cc, args.chosen_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = chosen_model(num_classes=1, n_channels=1, pretrained=True, freeze=True if args.freeze else False)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\tTransposed conv upscaling')

    if args.load:
        # load network params, optimzer params, scheduler params, last loss, last epoch
        checkpoint = torch.load(args.load, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"], strict=args.strict)
        net.freeze_vgg(True if args.freeze else False)
        current_epoch = checkpoint["current_epoch"]
        current_loss = checkpoint["current_loss"]
        current_best_val_score = checkpoint["current_best_val_score"]
        epoch_of_best_model = checkpoint["epoch_of_best_model"]

        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # best_model_state = deepcopy(net.state_dict())

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  last=args.last,
                  best=args.best,
                  log_dir=args.log_dir,
                  )
    except KeyboardInterrupt:
        pass

