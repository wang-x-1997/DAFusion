#coding= utf_8
from __future__ import print_function
# from scipy.misc import imread, imresize
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import time
import glob
from net_utilis import *
from net_utilis import Wav
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt



def RGB2Y(img):
    img = img
    r, g, b = torch.split(img, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y
def RGB2YCbCr(img):
    r, g, b = torch.split(img, 1, dim=1)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    return y, cb, cr

def YCbCr2RGB(y, cb, cr):

    cb = cb - 128
    cr = cr - 128

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    r = torch.clamp(r / 255.0, 0.0, 1.0)
    g = torch.clamp(g / 255.0, 0.0, 1.0)
    b = torch.clamp(b / 255.0, 0.0, 1.0)

    img = torch.cat([b, g, r], dim=1)
    return img*255

def restore_color(Y_img, RGB_img):

    _, Cb, Cr = RGB2YCbCr(RGB_img)

    restored_RGB = YCbCr2RGB(Y_img, Cb, Cr)
    return restored_RGB


def prepare_data(dataset):
    # data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data_dir =dataset
    data = glob.glob(os.path.join(data_dir, "IR*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.bmp")))
    # a = data[0][len(str(data_dir))+1:-6]
    data.sort(key=lambda x:int(x[len(str(data_dir))+2:-4]))
    return data
def prepare_data1(dataset):
    # data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data_dir = dataset
    data = glob.glob(os.path.join(data_dir, "VIS*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.bmp")))
    data.sort(key=lambda x:int(x[len(str(data_dir))+3:-4]))
    return data


def change(out):
    out1 = out.cpu()
    out_img = out1.data[0].numpy()
    out_img = out_img.transpose(1, 2, 0).astype(np.uint8)  # 确保数据类型为uint8
    return out_img

def change_gray(out):
    out1 = out.cpu()
    out_img = out1.data[0]
    out_img = out_img.squeeze()
    out_img = out_img.numpy()
    # out_img = out_img.transpose(1, 2, 0)
    return out_img
def count_parameters_in_MB(model):
    return (np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6)

def load_image_VIS(x):
  imgA = Image.open(x)
  # imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  imgA = imgA.unsqueeze(0)
  return imgA
def load_image_IR(x):
  imgA = Image.open(x)
  imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  imgA = imgA.unsqueeze(0)
  return imgA

def load_rgb(x):
  imgA = Image.open(x)
  # imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  imgA = imgA.unsqueeze(0)
  return imgA

def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x-min1)/(max1-min1 + 1e-10)

def name_list(path):
    filename_list = os.listdir(path)
    source_name = []
    save_name = []
    for i in filename_list:
        filename = str(i)
        b = filename.split('.')
        source_name.append(filename)
        save_name.append(b[0])
    return source_name,save_name


start=time.time()
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
device = torch.device("cuda")

model_path_E =r'./model/model_E.pth'
model_path_D =r'./model/model_D.pth'
model_path_F =r'./model/model_F.pth'

model_E = torch.load(model_path_E)
model_D = torch.load(model_path_D)
model_F = torch.load(model_path_F)

print(count_parameters_in_MB(model_D)+count_parameters_in_MB(model_E)+count_parameters_in_MB(model_F))
image_A_list = prepare_data(r'D:\Image_Data\IRVI\AUIF Datasets\16x\Test_FLIR/')
image_B_list = prepare_data1(r'D:\Image_Data\IRVI\AUIF Datasets\16x\Test_FLIR/')
save_path = "./Fused/"
if os.path.exists(save_path):
    pass
else:
    os.makedirs(save_path)
source_name,save_name = name_list(r'D:\Image_Data\IRVI\MSRS-main\MSRS-main\train\ir/')
all_time = []


for i in range(len(image_A_list)):
    S = time.time()
    imgB = load_image_IR(image_A_list[i])
    imgA = load_image_VIS(image_B_list[i])
    imgA_Y = RGB2Y(imgA)

    if not opt.cuda:
        model_E = model_E.to(device)
        model_D = model_D.to(device)
        model_F = model_F.to(device)
        imgA = (imgA).to(device)
        imgA_Y = (imgA_Y).to(device)
        imgB = (imgB).to(device)

    with torch.no_grad():

        F_ir, emb_ir, F_vis, emb_vis= model_E(imgA_Y,imgB)
        F,emb_F = model_F(F_ir,F_vis, [emb_ir,  emb_vis] )
        F,F_fea= model_D(F, emb_F)
        F = restore_color(F,imgA)
    out = change(F)
    cv2.imwrite(os.path.join(save_path, str(i+1) + '.bmp'), out)
    print('mask'+str(i+1)+' has saved')
    all_time.append(time.time() - S)

print('Mean [%f], var [%f]'% (np.mean(all_time), np.std(all_time)))


