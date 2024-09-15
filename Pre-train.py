# coding=utf-8
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pytorch_msssim
from TD import TD
from loss import *
import kornia
import matplotlib
import matplotlib.pyplot as plt
from dataset import pre_dataset
from net_utilis import *
import net

# Set backend for matplotlib to avoid display issues in headless environments
matplotlib.use('Agg')


def str2bool(v):
    if v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def RGB2Y(img):
    img = img
    r, g, b = torch.split(img, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def restore_ema(model, ema_model, alpha=0):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_ema_variables(model, ema_model, alpha, global_step):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(net, ema=False):
    model = net
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', type=str, default='data', help='Dataset path')
parser.add_argument('--batchSize', type=int, default=8, help='Training batch size')
parser.add_argument('--testBatchSize', type=int, default=8, help='Testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=1, help='Input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='Output image channels')
parser.add_argument('--ngf', type=int, default=64, help='Generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='Discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam. Default=0.5')
parser.add_argument('--cuda', action='store_true', help='Use CUDA?')
parser.add_argument('--threads', type=int, default=0, help='Number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed. Default=123')
parser.add_argument('--lamb', type=int, default=150, help='Weight on L1 term in objective')
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay rate')
opt = parser.parse_args()

use_cuda = opt.cuda and torch.cuda.is_available()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda" if use_cuda else "cpu")

print('===> Loading datasets')
root_path = "data/"
predataset = pre_dataset(os.path.join(root_path, opt.dataset))
training_predata_loader = DataLoader(dataset=predataset, num_workers=opt.threads, batch_size=opt.batchSize,
                                     shuffle=True)

print('===> Building model')
os.makedirs("./model/", exist_ok=True)

# Define loss functions and optimizer
Mse_loss = nn.MSELoss()
L1_loss = nn.L1Loss()
SL1_loss = nn.SmoothL1Loss()
SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')
TDloss = TD(device='cuda')
criterion = [SimMaxLoss(metric='cos', alpha=opt.alpha).to(device), SimMinLoss(metric='cos').to(device),
             SimMaxLoss(metric='cos', alpha=opt.alpha).to(device)]
wav_loss = Wav()

model_E_res = net.Dual_Encoder().to(device)
model_D_res = net.Decoder_res().to(device)

optimizer_E = optim.Adam(model_E_res.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer_D = optim.Adam(model_D_res.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.9)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_E, milestones=[15, 18], gamma=0.9)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[15, 18], gamma=0.9)

print('---------- Networks initialized -------------')
print('-----------------------------------------------')

loss_plot = []


def train(epoch):
    for iteration, batch in enumerate(training_predata_loader, 1):
        imgA, imgB = batch[0].to(device), batch[1].to(device)
        imgB = RGB2Y(imgB)

        _, _, h, w = imgA.shape

        F_ir, emb_ir, F_vis, emb_vis = model_E_res(imgA, imgB)
        IR = model_D_res(F_ir, emb_ir)
        VIS = model_D_res(F_vis, emb_vis)

        loss = torch.norm(IR - imgA, p=2) + 5 * SSIMLoss(IR, imgA) + torch.norm(VIS - imgB) + 5 * SSIMLoss(VIS, imgB)

        optimizer_E.zero_grad()
        optimizer_D.zero_grad()
        loss.backward()

        optimizer_E.step()
        optimizer_D.step()

        if epoch % 20 == 0:
            torch.save(model_E_res, f"./model/model_CE_res_{epoch}.pth")
            torch.save(model_D_res, f"./model/model_CD_res_{epoch}.pth")
            loss_plot.append(loss.item())

    scheduler.step()
    scheduler1.step()


if __name__ == '__main__':
    for epoch in range(1, 21):
        train(epoch)
        print(f'This is epoch: {epoch}')
    plt.plot(loss_plot)
    plt.show()
