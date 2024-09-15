import os
import numpy as np
import torch.utils.data as data1
import torchvision.transforms as transforms
# from scipy.misc import imread, imresize  #1.2.1
import torch
import random
from PIL import Image
import cv2

def load_image(x):
  imgA = Image.open(x)
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  return imgA

def load_rgb(x):
  imgA = Image.open(x)
  # imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  return imgA

def pre_dataset_path(root, train=True):
    dataset = []

    if train:
      dir_img = os.path.join(r'D:\Image_Data\RoadScene-master\256')
      for index in range(3383):
        imgA = str(index + 1) + '_1.jpg'
        imgB = str(index + 1) + '_2.jpg'
        dataset.append([os.path.join(dir_img, imgA), os.path.join(dir_img, imgB)])

    return dataset


class pre_dataset(data1.Dataset):

  def __init__(self, root, transform=None, train=True):
    self.train = train

    if self.train:
      self.train_set_path = pre_dataset_path(root, train)

  def __getitem__(self, idx):
    if self.train:
      imgA_path,imgB_path = self.train_set_path[idx]

      imgA = load_image(imgA_path)
      imgB = load_rgb(imgB_path)

      return imgA,imgB

  def __len__(self):
    if self.train:
      return 3383 # 9000



def make_dataset(root, train=True):
  dataset = []

  if train:

    dir_img = os.path.join(r'D:\Image_Data\Exposure_image\training_set\256-GT')
    for index in range(3383):
      imgA = str(index + 1) + '.png'
      dataset.append([os.path.join(dir_img, imgA)])

  return dataset


class fusiondata(data1.Dataset):

  def __init__(self, root, transform=None, train=True):
    self.train = train

    if self.train:
      self.train_set_path = make_dataset(root, train)

  def __getitem__(self, idx):
    if self.train:
      imgA_path= self.train_set_path[idx]
      imgA = load_rgb(imgA_path)
      return imgA

      
  def __len__(self):
    if self.train:
      return 3383

