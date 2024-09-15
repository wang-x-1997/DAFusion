# coding=utf-8
from __future__ import print_function
import argparse
import os
from dataset import fusiondata, pre_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
import net
# from net import process as ploss
import pytorch_msssim
from TD import TD
from loss import *
import kornia
# import torchvision.models as models
import matplotlib.pyplot as plt
from net_utilis import *

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
parser.add_argument('--seed', type=int, default=123, help='Random seed to use. Default=123')
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
training_predata_loader = DataLoader(dataset=predataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
dataset = fusiondata(os.path.join(root_path, opt.dataset))
training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model')

# Ensure model directory exists
os.makedirs("./model/", exist_ok=True)

model_CE = torch.load(r'./model/model_CE_res_20.pth')
model_E_res = torch.load(r'./model/model_CE_res_20.pth')
model_E_res_ema = torch.load(r'./model/model_CE_res_20.pth')
model_D_res = torch.load(r'./model/model_CD_res_20.pth')
model_FNet = net.Fusion_layer().to(device)
model_FNet_ema = create_model(net.Fusion_layer(), ema=True).to(device)

# Define loss functions and optimizer
Mse_loss = nn.MSELoss()
L1_loss = nn.L1Loss()
SL1_loss = nn.SmoothL1Loss()
SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')
TDloss = TD(device='cuda')
criterion = [SimMaxLoss(metric='cos', alpha=opt.alpha).to(device), SimMinLoss(metric='cos').to(device),
             SimMaxLoss(metric='cos', alpha=opt.alpha).to(device)]
wav_loss = Wav()

optimizer_E = optim.Adam(model_E_res.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer_F = optim.Adam(model_FNet.parameters(), lr=0.001, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_E, milestones=[5, 8], gamma=0.9)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_F, milestones=[5, 8], gamma=0.9)

print('---------- Networks initialized -------------')
print('-----------------------------------------------')

loss_plot = []
loss_plot1 = []
loss_plot2 = []

def loss_res(F, imgA, emb, emb_ema, emb_ir_ir, emb_ema_F):
    b, c, h, w = imgA.shape
    nom_bchw = b * c * h * w
    b1, c1, h1, w1 = emb.shape
    nom_emb = b1 * c1 * h1 * w1

    loss1 = (1 * wav_loss(imgA, F) + SSIMLoss(imgA, F))
    loss2 = torch.norm(emb.reshape(emb.size(0), -1) - emb_ir_ir.reshape(emb_ema.size(0), -1)) \
            + criterion[0](emb.reshape(emb.size(0), -1), emb_ema.reshape(emb_ema.size(0), -1)) \
            + criterion[1](emb.reshape(emb.size(0), -1), emb_ema_F.reshape(emb_ema.size(0), -1))
    return loss1 / nom_bchw + loss2 / nom_emb

def loss_res_F(F, imgA, emb, emb_ema, emb_ir_ir, emb_ema_F):
    b1, c1, h1, w1 = emb.shape
    nom_emb = b1 * c1 * h1 * w1

    loss2 = torch.norm(emb.reshape(emb.size(0), -1) - emb_ir_ir.reshape(emb_ema.size(0), -1)) \
            + criterion[0](emb.reshape(emb.size(0), -1), emb_ema.reshape(emb_ema.size(0), -1)) \
            + criterion[1](emb.reshape(emb.size(0), -1), emb_ema_F.reshape(emb_ema.size(0), -1))
    return loss2 / nom_emb

def train(e):
    for iteration, (batch1, batch2) in enumerate(zip(training_predata_loader, training_data_loader), 1):
        imgA, imgB, imgC = batch1[0].to(device), batch1[1].to(device), batch2[0].to(device)
        imgB = RGB2Y(imgB)
        imgC = RGB2Y(imgC)

        b, c, h, w = imgA.shape

        F_ir, emb_ir, F_vis, emb_vis = model_E_res(imgA, imgB)
        F_ir_c, emb_ir_c, F_vis_c, emb_vis_c = model_CE(imgC, imgC)

        IR, D_IR = model_D_res(F_ir, emb_ir)
        VIS, D_VIS = model_D_res(F_vis, emb_vis)

        _, emb_ir_ir, _, emb_vis_vis = model_E_res(IR, VIS)
        _, emb_ema_ir_c, _, emb_ema_vis_c = model_E_res_ema(imgA, imgB)

        loss1 = loss_res(IR, imgA, emb_ir, emb_ir_c, emb_ir_ir, emb_ema_ir_c)
        loss2 = loss_res(VIS, imgB, emb_vis, emb_vis_c, emb_vis_vis, emb_ema_vis_c)

        Fea, emb_F = model_FNet(F_ir, F_vis, [emb_ir, emb_vis])
        Fea_1, emb_F_1 = model_FNet_ema(F_ir, F_vis, [emb_ir, emb_vis])

        F, _ = model_D_res(Fea, emb_F)
        _, emb_ema_ir_F, _, emb_ema_vis_F = model_E_res(F, F)

        wen_1, wen_2 = en_w([F_ir, F_vis], [D_IR, D_VIS])
        w1, w2 = sqt(IR, VIS)
        w1 = norm_1(w1 + wen_1)
        w2 = norm_1(w2 + wen_2)

        loss3 = loss_res_F(F, IR, emb_F, emb_ir_c, emb_ema_ir_F, emb_F_1)
        loss4 = loss_res_F(F, VIS, emb_vis, emb_vis_c, emb_vis_c, emb_F_1)
        loss5 = 0.7 * (wav_loss(IR, F) + SSIMLoss(IR, F) + wav_loss(VIS, F) + SSIMLoss(VIS, F)) / (b * c * h * w) + (
                    1 * torch.norm(w1 * F - w1 * IR, p=2) + 1 * torch.norm(w2 * F - w2 * VIS, p=2)) / (b * c * h * w)

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        optimizer_E.zero_grad()
        optimizer_F.zero_grad()
        update_ema_variables(model_E_res, model_E_res_ema, opt.ema_decay, iteration)
        update_ema_variables(model_FNet, model_FNet_ema, opt.ema_decay, iteration)
        loss.backward()

        optimizer_E.step()
        optimizer_F.step()

    if e % 10 == 0:
        net_g_model_out_path1 = "./model/model_E_res_{}.pth".format(e)
        torch.save(model_E_res, net_g_model_out_path1)
        net_g_model_out_path2 = "./model/model_F_res_{}.pth".format(e)
        torch.save(model_FNet, net_g_model_out_path2)

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch + 1)
        print('This is epoch: ' + str(epoch + 1))
    # plt.plot(loss_plot)
    # plt.show()
