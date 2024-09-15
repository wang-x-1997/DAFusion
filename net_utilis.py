import torch
from torch import nn
import torch.nn.functional as F
import math
from function import adaptive_instance_normalization as adain
# from s import fused_leaky_relu
import numpy as np
from extractor import VitExtractor

def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x - min1) / (max1 - min1 + 1e-10)

import torch
from torch import nn
from torch.nn import functional as F


class Fusion_net(nn.Module):
    def __init__(self):
        super(Fusion_net, self).__init__()
        self.fc_ir = nn.Linear(64, 64)
        self.fc_vis = nn.Linear(64, 64)
        self.sig = nn.Sigmoid()
        self.Conv_embf = nn.Sequential(
            nn.Conv2d(128,64,3,1,1)
        )
        self.Conv_f = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self,ir,vis,emb_ir,emb_vis):
        ir_1,vis1 = self.CA_attention(ir ,vis,emb_ir,emb_vis)
        f,emb_f = self.SA_attention(ir_1,vis1,emb_ir,emb_vis)
        # f = self.Conv_f(torch.cat([ir,vis],1))
        # emb_f = self.Conv_embf(torch.cat([emb_ir,emb_vis],1))

        return f,emb_f

    def CA_attention(self,ir ,vis,emb_ir,emb_vis):
        ir_atten = self.sig(self.fc_ir(emb_ir.reshape(ir.size(0),-1)))
        vis_atten = self.sig(self.fc_vis(emb_vis.reshape(vis.size(0), -1)))
        ir_atten_f = ir_atten.reshape(ir.size(0),ir.size(1),1,1)*ir
        vis_atten_f = vis_atten.reshape(ir.size(0),ir.size(1),1,1) * vis
        return ir_atten_f,vis_atten_f

    def SA_attention(self,ir,vis,emb_ir,emb_vis):
        emb_f = self.Conv_embf(torch.cat([emb_ir,emb_vis],1))
        SA_emb = self.sig(emb_f)
        SA_ir = vis*SA_emb + ir
        SA_vis = vis + ir*SA_emb
        f = self.Conv_f(torch.cat([SA_ir,SA_vis],1))
        emb_ff = self.avg(f) + emb_f
        return f, emb_ff


def BNReLU(num_features):
    return nn.Sequential(
        nn.BatchNorm2d(num_features),
        nn.ReLU()
    )
def BatchNorm2d():
    return nn.BatchNorm2d



class HighClue_Guided_Intra_and_Cross_Domain_Refin(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, low_in_channels):
        super(HighClue_Guided_Intra_and_Cross_Domain_Refin, self).__init__()

        self.in_channels = low_in_channels
        self.out_channels = low_in_channels
        self.key_channels = low_in_channels
        self.value_channels = low_in_channels


        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(self.key_channels),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(self.key_channels),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        # self.conv = nn.Conv2d()
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, low_feats, heat_emb):
        batch_size, h, w = low_feats.size(0), low_feats.size(2), low_feats.size(3)

        heat_emb[0] = heat_emb[0].view(batch_size, self.key_channels, -1)
        heat_emb[0] = heat_emb[0].permute(0, 2, 1)# b,c,h*w -> b,h*w,c
        heat_emb[1] = heat_emb[1].view(batch_size, self.key_channels, -1)
        heat_emb[1] = heat_emb[1].permute(0, 2, 1)  # b,c,h*w -> b,h*w,c


        value = self.f_value(low_feats).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        value_I = heat_emb[0] * value
        value_C = heat_emb[1] * value

        # query = self.f_query(low_feats).view(batch_size, self.key_channels, -1)
        # query = query.permute(0, 2, 1) # b,h*w,c
        # key = self.f_key(low_feats) # b,c,h*w
        # sim_map = torch.matmul(query, key)
        # sim_map = F.softmax(sim_map, dim=-1)
        # context = torch.matmul(sim_map, value)
        # context = context.permute(0, 2, 1).contiguous()
        # context = context.view(batch_size, self.value_channels, *low_feats.size()[2:])

        sim_map = self.QK_map(low_feats)
        Fea_I = self.QK_V(sim_map,value_I,low_feats)
        Fea_V = self.QK_V(sim_map, value_C, low_feats)


        context = (Fea_I+Fea_V)
        return context

    def QK_map(self,low_feats):
        batch_size, h, w = low_feats.size(0), low_feats.size(2), low_feats.size(3)
        query = self.f_query(low_feats).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # b,h*w,c
        key = self.f_key(low_feats).view(batch_size, self.key_channels, -1)  # b,c,h*w

        sim_map = torch.matmul(query, key)
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map

    def QK_V(self,sim_map,value,low_feats):
        batch_size, h, w = low_feats.size(0), low_feats.size(2), low_feats.size(3)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *low_feats.size()[2:])
        return context


class Wav(nn.Module):
    def __init__(self):
        super(Wav, self).__init__()

    def forward(self,x,y):
        # _,x_lh,x_hl,x_hh = self.Wavconv(x)
        # _, y_lh, y_hl, y_hh = self.Wavconv(y)
        x_hh = self.Wavconv(x)
        y_hh = self.Wavconv(y)
        loss =0
        for i in range(1,len(x_hh)):
            loss += torch.norm(x_hh[i]-y_hh[i])
        return loss/3



    def get_wav(self,in_channels=1,out_channels=32):
        # """wavelet decomposition using conv2d"""
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
        filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
        filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
        filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

        net = nn.Conv2d

        LL = net(in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0, bias=False,
                 groups=in_channels)
        LH = net(in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0, bias=False,
                 groups=in_channels)
        HL = net(in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0, bias=False,
                 groups=in_channels)
        HH = net(in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0, bias=False,
                 groups=in_channels)

        LL.weight.requires_grad = False
        LH.weight.requires_grad = False
        HL.weight.requires_grad = False
        HH.weight.requires_grad = False

        LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
        LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
        HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
        HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

        return LL.cuda(), LH.cuda(), HL.cuda(), HH.cuda()

    def Wavconv(self,x):
        self.LL, self.LH, self.HL, self.HH = self.get_wav()
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)



class DINO_loss(nn.Module):
    def __init__(self):
        super(DINO_loss, self).__init__()
        self.extractor = VitExtractor()

    def forward(self,x,y):
        loss = self.calculate_global_ssim_loss(x,y)
        return loss

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0

        a =  torch.cat([outputs,outputs,outputs],1)
        b =  torch.cat([inputs,inputs,inputs],1)
        for i in range(a.size(0)):
            image_ir = a[i]
            image_vis = b[i]
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(image_ir.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(image_vis.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)

        return loss

def sumPatch(x,k):
    dim = x.shape
    kernel = np.ones((2*k+1,2*k+1))
    kernel = kernel/(1.0*(2*k+1)*(2*k+1))
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1)
    weight = nn.Parameter(data=kernel,requires_grad=False)
    weight = weight.cuda()
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=k)
    return gradMap

def sqt(F_ir,F_vis):
    F_1 = F_ir
    F_2 = F_vis
    # F_1 = torch.sqrt((F_1 - torch.mean(F_1)) ** 2)
    # F_2 = torch.sqrt((F_2 - torch.mean(F_2)) ** 2)
    # F_1 = torch.split(F_ir, 4, 1)
    # F_2 = torch.split(F_vis, 4, 1)
    F_1 = torch.sqrt((F_1 - torch.mean(F_1)) ** 2)
    F_2 = torch.sqrt((F_2 - torch.mean(F_2)) ** 2)
    g_ir = F_1.sum(dim=1, keepdim=True) / 64
    g_vi = F_2.sum(dim=1, keepdim=True) / 64
    w1 = g_ir.greater(g_vi)
    w2 = ~w1
    w1 = w1.to(torch.int)
    w2 = w2.to(torch.int)
    # w1 = g_ir / (g_vi + g_ir + 1e-10)
    # w2 = 1 - w1
    return  w1,w2

def act(F_ir,F_vis):
    F_1 = torch.split(F_ir, 4, 1)
    F_2 = torch.split(F_vis, 4, 1)
    # F_1 = F_ir
    # F_2 = F_vis
    g_ir = F_1[0].sum(dim=1, keepdim=True) / 16
    g_vi = F_2[0].sum(dim=1, keepdim=True) / 16
    g_ir = sumPatch(g_ir, 3)
    g_vi = sumPatch(g_vi, 3)
    w1 = g_ir.greater(g_vi)
    w2 = ~w1
    w1 = w1.to(torch.int)
    w2 = w2.to(torch.int)
    # w1 = g_ir/(g_vi+g_ir+1e-10)
    # w2 = 1-w1
    return w1,w2

def cc(tensor,tensor1):

    c, d, h, w = tensor.size()
    tensor = tensor.view(c, d * h*w)
    tensor1 = tensor1.view(c, d * h * w)
    gram = torch.mm(tensor , tensor1.t())
    return gram.mean()

def en_ac(x,y):
    F_1 = torch.split(x, 4, 1)
    F_1 = torch.cat([F_1[1], F_1[2], F_1[3]], 1)
    D_fea = torch.cat([y[0], y[1], y[2]], 1)
    c_d = D_fea.size(1)
    g_ir = D_fea.sum(dim=1, keepdim=True)/c_d
    g_vi = F_1.sum(dim=1, keepdim=True) / 48
    w1 = norm_1(g_ir - g_vi)
    return w1
def en_w(x,y):
    w1 = en_ac(x[0], y[0])
    w2 = en_ac(x[1], y[1])
    w1 = w1.greater(w2)
    w2 = ~w1
    return w1,w2

class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)
def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss