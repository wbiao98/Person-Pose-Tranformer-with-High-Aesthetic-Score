import argparse
import os
import torch
from model import modules


class PoseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self,parser):
        parser.add_argument('--load_size',type=int,default=256,help='Scale')
        parser.add_argument('--phase',type=str,default='train',help='train val test')
        parser.add_argument('--dataroot',type=str,default='./dataset/train',help='dataroot')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--angle', type=float, default=False)
        parser.add_argument('--shift', type=float, default=False)
        parser.add_argument('--scale', type=float, default=False)
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--old_size', type=int, default=(256,256), help='Scale images to this size.')
        return parser


    def parse(self):
        parser = self.initialize(self.parser)
        opt = parser.parse_args()
        self.opt = opt
        return self.opt
