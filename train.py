import os
os.environ["WANDB_API_KEY"] = "76ab78978f41b7190f2b6ca4a7a7836a27eb19ef"
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data import DataSet
from model import RECTNET
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR
import scipy.io as sio

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

def save_checkpoint(model, optimizer, epoch):  # save model function
    check_point = {'model': model.state_dict(), 
                   'optimizer': optimizer.state_dict(), 
                   'epoch': epoch
                   }
    save_path = 'rotate_final_simple' + '/' + f"{epoch}.pth"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(check_point, save_path)

def train(train_set, validate_set, config):
    epochs, lr, ckpt, batch_size, train_set_path, train_validate_path, checkpoint_path = \
        config.epochs, config.learning_rate, config.ckpt, config.batch_size, config.train_set_path, config.validate_set_path, config.checkpoint_path
    train_set = DataSet(train_set_path)
    validate_set = DataSet(train_validate_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                        pin_memory=True, drop_last=True)
    criterion = nn.MSELoss().to(device)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"=> successfully loaded checkpoint from '{checkpoint_path}'")
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
        #余弦衰减
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RECTNET().to(device)
        model = nn.DataParallel(model)
        epoch = 1

    print('Start training...')
    nx, ny = [0 for _ in range(10)], [0 for _ in range(10)]
    while epoch <= epochs + 1:
        if epoch == 101:
            tensor = [torch.tensor(sio.loadmat(f"x_{i}.mat")['x']).to(device) for i in range(1, 11)]
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
            output = model(pan, lms, epoch, *nx, *ny)
            loss = criterion(output, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        if epoch % ckpt == 0:
            save_checkpoint(model, optimizer, epoch)
        scheduler.step()  # 更新学习率
        print("lr: ", optimizer.param_groups[0]['lr'])
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, pan = batch[0].to(device), batch[1].to(device), batch[4].to(device)
                output = model(pan, lms, epoch, *nx, *ny)
                loss = criterion(output, gt)
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        print('validate loss: {:.7f}'.format(v_loss))
        f = open('loss_final_simple.txt', 'a')
        f.write(f'epoch: {epoch} | train_loss: {t_loss} | val_loss: {v_loss}\n')
        wandb.log({"train_loss": t_loss, "val_loss": v_loss})
        epoch = epoch + 1
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=1200, type=int, help="Total number of epochs.")
    parser.add_argument("--lr", default=0.0005, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--ckpt", default=20, type=int, help="Save model every ckpt epochs.")
    parser.add_argument("--train_set_path", default="training_data/train_wv3.h5", type=str, help="Path to the training set.")
    parser.add_argument("--validate_set_path", default="training_data/valid_wv3.h5", type=str, help="Path to the validation set.")
    parser.add_argument("--checkpoint_path", default="", type=str, help="Path to the checkpoint file.")
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