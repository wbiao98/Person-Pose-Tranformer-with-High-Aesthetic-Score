import argparse
import os
import torch
from model import modules

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--load_size', type=int, default=256, help='Scale')
        parser.add_argument('--phase', type=str, default='val', help='train val test')
        parser.add_argument('--dataroot', type=str, default='./dataset/val', help='dataroot')
        parser.add_argument('--angle', type=float, default=False)
        parser.add_argument('--shift', type=float, default=False)
        parser.add_argument('--scale', type=float, default=False)
        parser.add_argument('--lr', type=float, default=0.000001, help='initial learning rate for adam')
        parser.add_argument('--old_size', type=int, default=(256,256), help='Scale images to this size.')
        return parser

    def parse(self):
        parser = self.initialize(self.parser)
        opt = parser.parse_args()
        self.opt = opt
        return self.opt