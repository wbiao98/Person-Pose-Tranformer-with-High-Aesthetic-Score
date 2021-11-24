import torch
from torch import nn
from model.generator import GPoseNet
from model.discriminator import ResDiscriminator
from model import modules
from torch import optim
import numpy as np
from loss_utils import (
    AdversarialLoss,
    VGGLoss
)
from model.modules import (PoseDecoder,PoseEncoder)
from itertools import islice
import itertools
import pytorch_ssim
from LossParsing import ParsingLoss 

class TransposeNet(nn.Module):
    def __init__(self,opt,in_channels,out_channels,device=None):
        super(TransposeNet,self).__init__()
        self.loss_names = ['mse_gen','content_gen','style_gen','parsing_gen']
        self.device = device
        self.opt = opt
        self.netG = GPoseNet(in_channels,out_channels).to(device=device)

        self.optimizer_G = optim.RMSprop(self.netG.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
        #self.netD = ResDiscriminator().to(device=device) 
        #self.optimizer_D = optim.RMSprop(self.netD.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)

        #define the losss functions
        #self.L1loss = torch.nn.L1Loss()
        #self.GANloss = AdversarialLoss().to(device=device)
        self.Vggloss = VGGLoss().to(device=device)
        self.Parsingloss = ParsingLoss().to(device=device)
        self.Mseloss= nn.MSELoss()
        #self.ssim_loss = pytorch_ssim.SSIM()


    def forward(self):
        self.out_img = self.netG(self.input_P1BP1,self.input_BP1BP2) 


    def test(self):
        """Forward function used in test time"""
        img_gen = self.netG(self.input_P1BP1,self.input_BP1BP2)
        return img_gen
    
    def set_input(self,input):
        self.input = input
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']
        P1BP1 = torch.cat((input_P1, input_BP1), 1)
        BP1BP2 = torch.cat((input_BP1, input_BP2), 1)
        self.input_P1BP1=P1BP1.to(device=self.device)                
        self.input_BP1BP2 = BP1BP2.to(device=self.device)          
        self.input_P2 = input_P2.to(device=self.device)

    def backward_GANloss_net(self,netD,real,fake):
    
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real,True)
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake,False)
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D_loss.backward()

        return D_loss

    def backward_D(self):
        modules._unfreeze(self.netD)
        self.loss_img_gen = self.backward_GANloss_net(self.netD,self.input_P2,self.out_img)

    def backward_G(self):
        # Calculate l1 loss
        #out_img = self.out_img
        #loss_app_gen = self.L1loss(self.out_img,self.input_P2)

        #self.loss_app_gen = loss_app_gen* self.opt.lambda_rec

        # Calculate GAN loss
        #modules._freeze(self.netD)
        #D_fake = self.netD(self.out_img)
        
        
        
        self.loss_mse_gen = self.Mseloss(self.out_img,self.input_P2)
        
        
        
        
        #self.loss_ad_gen = self.GANloss(D_fake,True)*self.opt.lambda_g

        # Calculate perceptual loss
        input_p2 = self.input_P2
        loss_content_gen, loss_style_gen = self.Vggloss(self.out_img,self.input_P2) 
        self.loss_style_gen = loss_style_gen * self.opt.lambda_style
        self.loss_content_gen = loss_content_gen * self.opt.lambda_content
        
        # Calculate parsing loss
        
        self.loss_parsing_gen = self.Parsingloss(self.out_img,self.input_P2)
        
        #self.loss_ssim_out = 1.0-self.ssim_loss(self.out_img, input_p2)
        
        total_loss = 0

        for name in self.loss_names:
            total_loss += getattr(self, "loss_" + name)
        total_loss.backward()
        print("total_lose",total_loss.item(),"loss_mse",self.loss_mse_gen.item(),"loss_style_gen",self.loss_style_gen.item(),"loss_content_gen",self.loss_content_gen.item(),"loss_parsing_gen",self.loss_parsing_gen.item())


    def optimize_parameters(self):
        self.forward()
        #self.optimizer_D.zero_grad()
        #self.backward_D()
        #self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_optimize(self):
        return self.optimizer_G.state_dict()

    

if __name__ == '__main__':
    net = TransposeNet(in_channels=6,out_channels=3)
    print(net)
