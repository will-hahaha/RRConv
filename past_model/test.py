import torch
import os
from scipy import io as sio
from torgb import to_rgb
from model_final_simple import RECTNET
import numpy as np
import h5py
import torch.nn as nn
from matplotlib import pyplot as plt
import cv2
from einops import rearrange
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

eras_list = []
sam_list = []
img_path = 'result/'

def get_sam(im1, im2):
    eps = 2.2204e-16
    im1 = im1.permute(1, 2, 0).detach().cpu().numpy()
    im2 = im2.permute(1, 2, 0).detach().cpu().numpy()
    assert im1.shape == im2.shape
    H, W, C = im1.shape
    im1 = np.reshape(im1, (H * W, C))
    im2 = np.reshape(im2, (H * W, C))
    core = np.multiply(im1, im2)
    mole = np.sum(core, axis=1)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    sam = np.rad2deg(np.arccos(((mole + eps) / (deno + eps)).clip(-1, 1)))
    return np.mean(sam)

def get_ergas(reference_image, processed_image):
    # 获取图像的尺寸
    reference_image = reference_image.permute(1, 2, 0).detach().cpu().numpy()
    processed_image = processed_image.permute(1, 2, 0).detach().cpu().numpy()
    height, width, num_bands = reference_image.shape
    # 初始化变量用于计算各个波段的MSE和RMSE
    mse_values = []
    rmse_values = []
    for band in range(num_bands):
        # 计算MSE（均方误差）
        mse = np.mean((reference_image[:, :, band] - processed_image[:, :, band]) ** 2)
        mse_values.append(mse)
        # 计算RMSE（均方根误差）
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)
    # 计算每个波段的平均亮度
    average_brightness = [np.mean(reference_image[:, :, band]) for band in range(num_bands)]
    # 定义常数参数
    N = num_bands
    L = 256  # 假设灰度级数为256
    # 计算ERGAS
    ergas_values = []
    for mse, rmse, Y in zip(mse_values, rmse_values, average_brightness):
        ergas_values.append((100 / L) * np.sqrt(1 / mse) * (rmse / Y) ** 2)
    ergas = np.sqrt((1 / N) * np.sum(ergas_values))
    return ergas

def load_set(file_path):
    data = h5py.File(file_path)
    gt = torch.from_numpy(np.array(data['gt'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4]).float()
    lms = torch.from_numpy(np.array(data['lms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4]).float()
    ms = torch.from_numpy(np.array(data['ms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4]).float()
    pan = torch.from_numpy(np.array(data['pan'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4]).float()
    return gt, lms, ms, pan

def showimg(img, string, i):
    img = to_rgb(img)
    plt.imshow(img)
    plt.imsave(img_path + string + '_' + str(i) + '.png', img)
    plt.title(f"epoch: {string}")
    plt.show()

model = RECTNET().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('rotate/50.pth'))
model.eval()
file_path = "test_data/test_wv3_multiExm1.h5"
test_gt, test_lms, test_ms, test_pan = load_set(file_path)

for i in range(0, 20):
    with torch.no_grad():
        output3 = model(test_pan[i], test_lms[i])
        # showimg(output3[0][...][...][...], "output", i)
        # showimg(test_lms[0][0][...][...], "lms", i)
        # showimg(test_pan[0][0][...][...], "pan", i)
        # showimg(test_ms[0][0][...][...], "ms", i)
        # showimg(test_gt[0][0][...][...], "gt", i)
        ERAS = get_ergas(test_gt[i][0][0:3][...][...], output3[0][0:3][...][...])
        SAM = get_sam(test_gt[i][0][0:3][...][...], output3[0][0:3][...][...])
        eras_list.append(ERAS)
        sam_list.append(SAM)
        # output3 = output3 * 2047
        # test_gt = test_gt * 2047
        # test_lms = test_lms * 2047
        # test_ms = test_ms * 2047
        # test_pan = test_pan * 2047
        output3 = rearrange(output3[0], 'c h w -> h w c') * 2047
        save_name = os.path.join("test_results", 'output_mulExm_' + str(i) + '.mat')
        sio.savemat(save_name, {'sr': output3.cpu().numpy()})
        # save_name = os.path.join("test_results", 'rectnet_gt' + str(i) + '.mat')
        # sio.savemat(save_name, {'gt': np.array(test_gt.cpu())})
        # save_name = os.path.join("test_results", 'rectnet_pan' + str(i) + '.mat')
        # sio.savemat(save_name, {'pan': np.array(test_pan.cpu())})
        # save_name = os.path.join("test_results", 'rectnet_lms' + str(i) + '.mat')
        # sio.savemat(save_name, {'lms': np.array(test_lms.cpu())})
        # save_name = os.path.join("test_results", 'rectnet_ms' + str(i) + '.mat')
print("ERAS:{}".format(sum(eras_list) / len(eras_list)))
print("SAM:{}".format(sum(sam_list) / len(sam_list)))
