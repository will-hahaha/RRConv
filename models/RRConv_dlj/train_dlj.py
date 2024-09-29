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
from torch.optim.lr_scheduler import StepLR
import scipy.io as sio
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter  # 导入 tensorboard

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True


def freeze_bn_layers(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.eval()  # 让BN层进入评估模式，不更新均值和方差
            for param in module.parameters():
                param.requires_grad = False  # 冻结BN层的所有参数


def save_checkpoint(model, optimizer, epoch, scheduler):  # save model function
    check_point = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch,
                   'scheduler': scheduler.state_dict()
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
        epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> successfully loaded checkpoint from '{checkpoint_path}'")

    writer = SummaryWriter(log_dir='tensorboard_logs')  # 设置tensorboard日志路径
    #    optimizer.param_groups[0]['lr'] = 0.0004
    #    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    print('Start training...')
    nx, ny = [0 for _ in range(10)], [0 for _ in range(10)]
    while epoch <= epochs + 1:
        if epoch == 1000:
            freeze_bn_layers(model)
        if epoch == 101:
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
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        writer.add_scalar('Loss/train', t_loss, epoch)
        if epoch % ckpt == 0:
            save_checkpoint(model, optimizer, epoch, scheduler)
        scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])
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
        print('validate loss: {:.7f}'.format(v_loss))
        f = open('loss_ms.txt', 'a')
        f.write(f'epoch: {epoch} | train_loss: {t_loss} | val_loss: {v_loss}\n')
        writer.add_scalar('Loss/val', v_loss, epoch)
        epoch = epoch + 1
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=1200, type=int, help="Total number of epochs.")
    parser.add_argument("--lr", default=0.0004, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--ckpt", default=5, type=int, help="Save model every ckpt epochs.")
    parser.add_argument("--train_set_path", default="/Data2/DataSet/pansharpening/training_data/train_wv3.h5", type=str,
                        help="Path to the training set.")
    parser.add_argument("--validate_set_path", default="/Data2/DataSet/pansharpening/validation_data/valid_wv3.h5",
                        type=str, help="Path to the validation set.")
    parser.add_argument("--checkpoint_path", default="", type=str, help="Path to the checkpoint file.")
    parser.add_argument("--training_model", default="RRConv", type=str, help="Model to train.")
    config = parser.parse_args()
    train(config)
