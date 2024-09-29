import torch
import os
from scipy import io as sio
from utils.torgb import to_rgb
from model_for_test_wxy import RECTNET
import numpy as np
import h5py
import torch.nn as nn
from matplotlib import pyplot as plt
from einops import rearrange
import matlab.engine

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
    reference_image = reference_image.permute(1, 2, 0).detach().cpu().numpy()
    processed_image = processed_image.permute(1, 2, 0).detach().cpu().numpy()
    height, width, num_bands = reference_image.shape
    mse_values = []
    rmse_values = []
    for band in range(num_bands):
        mse = np.mean((reference_image[:, :, band] - processed_image[:, :, band]) ** 2)
        mse_values.append(mse)
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)
    average_brightness = [np.mean(reference_image[:, :, band]) for band in range(num_bands)]
    N = num_bands
    L = 256
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
model.load_state_dict(torch.load('pathwxy/checkpoint_600_2024-09-28-08-54-13.pth')['model'])
model.eval()
file_path = "test_data/test_wv3_multiExm1.h5"
test_gt, test_lms, test_ms, test_pan = load_set(file_path)
if  os.path.exists("wxymats/x_1.mat"):
    print("Loading wxymats")
    nx, ny = [0 for _ in range(10)], [0 for _ in range(10)]
    tensor = [torch.tensor(sio.loadmat(f"wxymats/x_{i}.mat")['x']).cuda() for i in range(1, 11)]
    for i in range(10):
        nx[i], ny[i] = tensor[i].split([1, 1], dim=-1)
        nx[i] = int(nx[i])
        ny[i] = int(ny[i])
for i in range(0, 20):
    with torch.no_grad():
        print('------------------------------------------------------')
        print(f'Processing image {i + 1} of 20')
        print('------------------------------------------------------')
        if os.path.exists("wxymats/x_1.mat"):
            print("Using wxymats")
            output3 = model(test_pan[i], test_lms[i], 1000, *nx, *ny)
        else:
            print("Not Using wxymats")
            output3 = model(test_pan[i], test_lms[i], 10)
        ERAS = get_ergas(test_gt[i][0][0:3][...][...], output3[0][0:3][...][...])
        SAM = get_sam(test_gt[i][0][0:3][...][...], output3[0][0:3][...][...])
        eras_list.append(ERAS)
        sam_list.append(SAM)
        output3 = rearrange(output3[0], 'c h w -> h w c') * 2047
        save_name = os.path.join(
            "D:\\Desktop\\Common\\AdaptativeConvolution\\MetricCode\\2_DL_Result\\PanCollection\\WV3_Reduced\\RectNet\\results",
            'output_mulExm_' + str(i) + '.mat')
        sio.savemat(save_name, {'sr': output3.cpu().numpy()})
print("ERAS:{}".format(sum(eras_list) / len(eras_list)))
print("SAM:{}".format(sum(sam_list) / len(sam_list)))
eng = matlab.engine.start_matlab()
eng.run(r'D:\Desktop\Common\AdaptativeConvolution\MetricCode\Demo1_Reduced_Resolution_MultiExm.m', nargout=0)
eng.quit()
print('\n\n\n')
file_path = r'D:\Desktop\Common\AdaptativeConvolution\MetricCode\test_wv3_multiExm1_Avg_RR_Assessment.tex'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    last_three_lines = lines[-3:]
for line in last_three_lines:
    print(line.strip())
