import logging
import os
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data import DataSet
from models.RRConv.model import RECTNET
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR
import scipy.io as sio
from torch.optim.lr_scheduler import ReduceLROnPlateau
os.environ["WANDB_API_KEY"] = "76ab78978f41b7190f2b6ca4a7a7836a27eb19ef"

# Set up logging
log_filename = f'training_log_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename),
                              logging.StreamHandler()])  # Log to both file and console
logger = logging.getLogger()

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

def save_checkpoint(model, optimizer, scheduler, epoch):  # save model function
    check_point = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'epoch': epoch
                   }
    save_path = 'checkpoints' + '/' + f"checkpoint_{epoch}_" + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                             time.localtime()) + ".pth"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(check_point, save_path)

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs, lr, ckpt, batch_size, train_set_path, train_validate_path, checkpoint_path, training_model = \
        config.epochs, config.lr, config.ckpt, config.batch_size, config.train_set_path, config.validate_set_path, config.checkpoint_path, config.training_model
    train_set = DataSet(train_set_path)
    validate_set = DataSet(train_validate_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)

    criterion = nn.MSELoss().to(device)
    model = RECTNET().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=100, gamma=0.85)
    epoch = 1
    model = nn.DataParallel(model)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        logger.info(f"=> successfully loaded checkpoint from '{checkpoint_path}'")

#    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999))
	
    logger.info('Start training...')
    optimizer.param_groups[0]['lr'] = 0.0004
    epochs = 1500
    nx, ny = [0 for _ in range(10)], [0 for _ in range(10)]
    while epoch <= epochs + 1:
        if epoch == 840:
            tensor = [torch.tensor(sio.loadmat(f"models_mats/x_{i}.mat")['x']).to(device) for i in range(1, 11)]
            for i in range(10):
                nx[i], ny[i] = tensor[i].split([1, 1], dim=-1)
                nx[i] = int(nx[i])
                ny[i] = int(ny[i])
        epoch_train_loss, epoch_val_loss = [], []
        model.train()
        pbar = tqdm(enumerate(training_data_loader), total=len(training_data_loader),
                    bar_format="{l_bar}{bar:10}{r_bar}")
        for iteration, batch in pbar:
            with torch.no_grad():
                gt = batch[0].to(device)
            lms = batch[1].to(device)
            pan = batch[4].to(device)
            optimizer.zero_grad()
            if training_model == 'RRConv':
                output = model(pan, lms, epoch, *nx, *ny)
            else:
                output = model(pan, lms)
            loss = criterion(output, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        logger.info('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))
        if epoch % ckpt == 0:
            save_checkpoint(model, optimizer, scheduler, epoch)
        scheduler.step()
        logger.info("lr: {}".format(optimizer.param_groups[0]['lr']))
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, pan = batch[0].to(device), batch[1].to(device), batch[4].to(device)
                if training_model == 'RRConv':
                    output = model(pan, lms, epoch, *nx, *ny)
                else:
                    output = model(pan, lms)
                loss = criterion(output, gt)
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        logger.info('validate loss: {:.7f}'.format(v_loss))
        with open('loss_ms.txt', 'a') as f:
            f.write(f'epoch: {epoch} | train_loss: {t_loss} | val_loss: {v_loss}\n')
        wandb.log({"train_loss": t_loss, "val_loss": v_loss})
        epoch += 1
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=1200, type=int, help="Total number of epochs.")
    parser.add_argument("--lr", default=0.0006, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--ckpt", default=20, type=int, help="Save model every ckpt epochs.")
    parser.add_argument("--train_set_path", default="training_data/train_wv3.h5",
                        type=str, help="Path to the training set.")
    parser.add_argument("--validate_set_path", default="training_data/valid_wv3.h5",
                        type=str, help="Path to the validation set.")
    parser.add_argument("--checkpoint_path", default="checkpoints/checkpoint_840_2024-09-11-08-43-25.pth", type=str,
                        help="Path to the checkpoint file.")
    parser.add_argument("--training_model", default="RRConv", type=str, help="Model to train.")
    config = parser.parse_args()
    wandb.login()
    wandb.init(
        project="RRConv",
        config={
            "learning_rate": config.lr,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "architecture": "RectNet",
            "dataset": "WV3"
        }
    )
    train(config)
