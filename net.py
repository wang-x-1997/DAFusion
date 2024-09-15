import torch
from torch import nn
import os
from function import adaptive_instance_normalization as adain
from net_utilis import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        # nn.Linear(64, 64),
                        # nn.LeakyReLU(0.1, True),
                        # nn.Linear(64, 256),
                    )

    def forward(self, x):

        E1_1 = self.con1(x)
        E2_1 = self.con2(E1_1)
        E3_1 = self.con3(torch.cat([E2_1, E1_1], 1))
        E4_1 = self.con4(torch.cat([E3_1, E2_1, E1_1], 1))
        Ef = torch.cat([E4_1,E3_1, E2_1, E1_1], 1)
        emb = self.mlp(Ef)

        return Ef, emb

class Dual_Encoder(nn.Module):
    def __init__(self):
        super(Dual_Encoder, self).__init__()
        self.encoder_ir = Encoder()
        self.encoder_vis = Encoder()

    def forward(self,ir,vis):
        F_ir,emb_ir = self.encoder_ir(ir)
        F_vis, emb_vis = self.encoder_vis(vis)
        return F_ir,emb_ir,F_vis, emb_vis

class Adain_module(nn.Module):
    def __init__(self,fea_in_c,fea_out_c,m=-0.8):
        super(Adain_module, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(fea_in_c,fea_out_c,3,1,1),
            nn.BatchNorm2d(fea_out_c),
            nn.ReLU(),
        )
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.sig = nn.Sigmoid()
        self.mlp = nn.Sequential(
                        nn.Conv2d(64, fea_in_c,3,1,1),
                        nn.LeakyReLU(),
                        nn.Conv2d(fea_in_c, fea_in_c,1,1,0)
                    )
    def forward(self,x,style):
        style_mlp = self.mlp(style)
        s_f = adain(x,style_mlp)
        w = self.sig(self.w)
        f = self.conv(s_f*w+x*(1-w))
        return f

class Decoder_res(nn.Module):
    def __init__(self,m=-0.8):
        super(Decoder_res, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.sig = nn.Sigmoid()
        self.decon1 = Adain_module(64,32)
        self.decon2 = Adain_module(32,16)

        self.decon3 = nn.Sequential(

            nn.Conv2d(16,1,3,1,1),
            nn.Tanh()
        )
        self.mlp = nn.Sequential(
                        nn.Conv2d(64, 16,3,1,1),
                        nn.LeakyReLU(),
                        nn.Conv2d(16, 16,1,1,0)
                    )

    def forward(self,x,emb):
        x1 = self.decon1(x,emb)
        x2 = self.decon2(x1,emb)
        emb_mlp = self.mlp(emb)
        x3 = adain(x2,emb_mlp)

        w = self.sig(self.w)
        F = self.decon3(x3*w + x2*(1-w))
        return F*255, [x1,x2,x3]


class Fusion_layer(nn.Module):
    def __init__(self):
        super(Fusion_layer, self).__init__()
        self.Fusion_net = Fusion_net()

        # self.decoder = Decoder_res()
    def forward(self,ir,vis,embs):
        F_fea, emb = self.Fusion_net(ir,vis,embs[0],embs[1])

        return F_fea,emb


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


###### Degration ######
#
# from moco.builder import MoCo
#
# class MOCO_Encoder(nn.Module):
#     def __init__(self):
#         super(MOCO_Encoder, self).__init__()
#
#         self.E = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1, True),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.mlp = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.LeakyReLU(0.1, True),
#             nn.Linear(256, 256),
#         )
#
#     def forward(self, x):
#         fea = self.E(x).squeeze(-1).squeeze(-1)
#         out = self.mlp(fea)
#
#         return fea, out
#
# class MOCO(nn.Module):
#     def __init__(self):
#         super(MOCO, self).__init__()
#
#         self.E = MoCo(base_encoder=MOCO_Encoder)
#
#     def forward(self, A,B):
#         if self.training:
#             x_query = A                       # b, c, h, w
#             x_key = B                            # b, c, h, w
#             # degradation-aware represenetion learning
#             fea, logits, labels = self.E(x_query, x_key)
#             return fea,logits, labels
#         else:
#             # degradation-aware represenetion learning
#             fea = self.E(A, B)
#             return fea
#
# ############################################################################
# from SimCLR.resblock_simclr import Resb_SimCLR
#
# class BlindSR(nn.Module):
#     def __init__(self, args):
#         super(BlindSR, self).__init__()
#
#         # Generator
#
#         # self.G = SFTMD()
#
#         # Encoder
#         self.E = MoCo(base_encoder=Encoder)
#         self.E = Resb_SimCLR(encoder=MOCO_Encoder())
#
#     def forward(self, lr_d_i, lr_d_j):
#         if self.training:
#             x_query = lr_d_i                          # b, c, h, w
#             x_key = lr_d_j                           # b, c, h, w
#
#             # degradation-aware represenetion learning
#             h_i, h_j, mlp_i, mlp_j = self.E(x_query, x_key)
#
#             # degradation-aware SR
#             #h_lr, _, _, _ = self.E(lr, lr)
#             sr_i = self.GE(lr_d_i, h_i)
#
#             return sr_i, mlp_i, mlp_j
#         else:
#             # degradation-aware represenetion learning
#             h_lr_i, _, _, _ = self.E(lr_d_i, lr_d_i)
#
#             # degradation-aware SR
#             sr = self.GE(lr_d_i, h_lr_i)
#
#             return sr


