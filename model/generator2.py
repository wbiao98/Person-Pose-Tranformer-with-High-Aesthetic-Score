import torch
from torch import nn
from model.modules import (
    UpBasicBlock,
    Up,
    DownBasicBlock,
    Down,
    Inconv,
    LastLayer,
    OutConv,
    Upcbam
)
import numpy as np
from model.modules import (PoseDecoder,PoseEncoder)
class GPoseNet(nn.Module):
    def __init__(self,in_channels,out_channels,device=None):
        super(GPoseNet, self).__init__()
        self.device =device
        self.inconv = Inconv(in_channels, 64)
        self.down11 = Down(64, 128)
        self.down12 = Down(128, 256)
        self.down13 = Down(256, 512)

        # BP1 BP2
        self.inconv2 = Inconv(36, 64)
        self.down21 = Down(64, 128)
        self.down22 = Down(128, 256)
        self.down23 = Down(256, 512)

        # Decoder
        self.last = LastLayer(1024, 512)
        self.up1 = Upcbam(512, 256)
        self.up2 = Upcbam(256, 128, False)
        self.up3 = Upcbam(128, 64, False)
        self.out = OutConv(64, 3)

        # self.Encoder1 = PoseEncoder(in_channels).cuda(0)
        # self.Encoder2 = PoseEncoder(in_channels=36).cuda(0)
        # self.Decoder = PoseDecoder(1024).cuda(1)



    def forward(self,P1BP1,BP1BP2):
        # x,x2,x3,x4 = self.Encoder1(P1BP1) # gpu 0
        # y = self.Encoder2(BP1BP2) # gpu 0
        # z = self.Decoder(x,y,x2,x3,x4) # gpu 1
        # return z
        x1 = self.inconv(P1BP1)
        x2 = self.down11(x1)
        x3 = self.down12(x2)
        x4 = self.down13(x3)
        xx1 = self.inconv2(BP1BP2)
        xx2 = self.down21(xx1)
        xx3 = self.down22(xx2)
        xx4 = self.down23(xx3)
        x5 = self.last(x4,xx4)
        x = self.up1(x5,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)
        out_img = self.out(x)

        return out_img

