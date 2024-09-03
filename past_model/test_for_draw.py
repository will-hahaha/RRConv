import torch
import os
from torgb import to_rgb
from model_finall import RECTNET
import numpy as np
import h5py
import torch.nn as nn
from matplotlib import pyplot as plt
import scipy.io as sio
import matplotlib.patches as patches

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
img_path = 'result_draw/'

def load_set(file_path):
    data = h5py.File(file_path)
    gt = torch.from_numpy(np.array(data['gt'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    lms = torch.from_numpy(np.array(data['lms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    ms = torch.from_numpy(np.array(data['ms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    pan = torch.from_numpy(np.array(data['pan'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    return gt, lms, ms, pan

def showimg(img, string, i):
    img = to_rgb(img)
    plt.imshow(img)
    plt.imsave(img_path + string + '_' + str(i) + '.png', img)
    plt.title(f"epoch: {string}")
    plt.show()

def get_rotated_rect(center, length, width, angle):
    l = length / 2
    w = width / 2
    pts = np.array([
        [-l, -w],
        [l, -w],
        [l, w],
        [-l, w]
    ])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_pts = np.dot(pts, rotation_matrix.T) + center
    return rotated_pts

def convert_to_rgb(image):
    # 取前三个通道
    if image.shape[0] >= 3:
        return image[:3]
    # 如果通道数少于3，则重复第一个通道
    return np.tile(image[0:1], (3, 1, 1))

model = RECTNET().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('rotate/520.pth'))
model.eval()

file_path = "test_data/test_wv3_multiExm1.h5"
test_gt, test_lms, test_ms, test_pan = load_set(file_path)

# 加载 offset 和 theta 张量
offset = torch.tensor(sio.loadmat("offset.mat")['offset'])
theta = torch.tensor(sio.loadmat("theta.mat")['theta'])
x = torch.tensor(sio.loadmat("x.mat")['x'])

with torch.no_grad():
    # output3 = model(test_pan[17], test_lms[17])
    x_np = x.cpu().numpy()[0]

    # 将256通道图像转换为3通道RGB图像
    rgb_image = convert_to_rgb(x_np)
    img = to_rgb(rgb_image)

    height, width = img.shape[:2]

    # 随机挑选1000个像素位置
    num_points = 1000
    random_points = np.random.randint(0, height * width, num_points)
    random_coords = [(pt // width, pt % width) for pt in random_points]

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # 遍历随机挑选的像素位置
    for i, j in random_coords:
        l = offset[0, 0, i, j].item()  # 获取长度
        w = offset[0, 9, i, j].item()  # 获取宽度
        angle = theta[0, 0, i, j].item()  # 获取旋转角度

        # 获取旋转矩形的中心和角度
        center = (j, i)

        # 获取旋转矩形的顶点
        rotated_pts = get_rotated_rect(center, l, w, angle)

        # 创建多边形
        polygon = patches.Polygon(rotated_pts, closed=True, edgecolor='r', facecolor='none', linewidth=1)

        # 添加到图像上
        ax.add_patch(polygon)

    # 显示并保存图像
    plt.title("Output Image with Rectangles")
    plt.axis('off')
    plt.savefig(img_path + 'output_with_rectangles_520.png')
    plt.show()
