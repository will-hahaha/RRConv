import os

os.environ["WANDB_API_KEY"] = "76ab78978f41b7190f2b6ca4a7a7836a27eb19ef"
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import DataSet
from model_final_simple import RECTNET
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR

wandb.login()
wandb.init(
    project="RRConv_simple",
    config={
        "learning_rate": 0.001,
        "epochs": 1200,
        "batch_size": 32,
        "architecture": "RectNet",
        "dataset": "WV3"
    }
)

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

lr = 0.001
epochs = 1200
ckpt = 20
batch_size = 32


def initialize_weights(module):
    # 初始化 Conv2d 层
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0, std=0.1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    # 初始化 Conv3d 层
    elif isinstance(module, nn.Conv3d):
        nn.init.normal_(module.weight, mean=0, std=0.1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


model = RECTNET().cuda()
model = nn.DataParallel(model)
# model.apply(initialize_weights)

model_path = "rotate_final_simple/40.pth"
startepoch = 1

if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    print("=> loaded checkpoint '{}'".format(model_path))
    startepoch = 41

criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = StepLR(optimizer, step_size=100, gamma=0.85)


# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=30)

def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'rotate_final_simple' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


def train(training_data_loader, validate_data_loader):
    print('Start training...')
    for epoch in range(startepoch, epochs + 1):
        epoch_train_loss, epoch_val_loss = [], []
        model.train()
        pbar = tqdm(enumerate(training_data_loader), total=len(training_data_loader),
                    bar_format="{l_bar}{bar:10}{r_bar}")
        for iteration, batch in pbar:
            # for iteration, batch in enumerate(training_data_loader, 1):
            with torch.no_grad():
                gt = batch[0].cuda()
            lms = batch[1].cuda()
            pan = batch[4].cuda()
            optimizer.zero_grad()
            output = model(pan, lms)
            loss = criterion(output, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        if epoch % ckpt == 0:
            save_checkpoint(model, epoch)
        scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, pan = batch[0].cuda(), \
                    batch[1].cuda(), \
                    batch[4].cuda()
                output = model(pan, lms)
                loss = criterion(output, gt)
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        print('validate loss: {:.7f}'.format(v_loss))
        f = open('loss_final_simple.txt', 'a')
        f.write('epoch: {} | train_loss: {} | val_loss: {}\n'.format(epoch, t_loss, v_loss))
        wandb.log({"train_loss": t_loss, "val_loss": v_loss})

    wandb.finish()


if __name__ == '__main__':
    train_set = DataSet('training_data/train_wv3.h5')
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_set = DataSet('training_data/valid_wv3.h5')
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    train(training_data_loader, validate_data_loader)