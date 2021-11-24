import glob
import torch
import torch.utils.data as data
import random
import os
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F
import pandas as pd
import torch
import math
import numbers
from util import util_pose
import numpy as np

class Pose_dataLoder(data.Dataset):
    def __init__(self):
        super(Pose_dataLoder, self).__init__()
        #self.data_path = data_path
        #self.imgs_path = glob.glob(os.path.join(data_path,'image/*.png'))
        
    def initialize(self,opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.name_pairs = self.get_path(opt)
        size = len(self.name_pairs)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        transform_list = []
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')
    
    def __getitem__(self, item):
        #image_path = self.imgs_path[item]
        P1_name, P2_name = self.name_pairs[item]
        P1_path = os.path.join(self.image_dir, P1_name)  # person 1
        P2_path = os.path.join(self.image_dir, P2_name)  # person 2

        P1_img = Image.open(P1_path).convert('RGB')
        
        P2_img = Image.open(P2_path).convert('RGB')

        P1_img = F.resize(P1_img, self.load_size)
        P2_img = F.resize(P2_img, self.load_size)

        # angle, shift, scale = self.getRandomAffineParam()
        # P1_img = F.affine(P1_img, angle=angle, translate=shift, scale=scale, shear=0, fill=(128, 128, 128))
        # center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        #affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        BP1 = self.obtain_bone(P1_name, affine_matrix=None)
        P1 = self.trans(P1_img)

        # angle, shift, scale = self.getRandomAffineParam()
        # angle, shift, scale = angle * 0.2, (
        # shift[0] * 0.5, shift[1] * 0.5), 1  # Reduce the deform parameters of the generated image
        # P2_img = F.affine(P2_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        # center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        #affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        BP2 = self.obtain_bone(P2_name, affine_matrix=None)
        P2 = self.trans(P2_img)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name}

    def get_path(self,opt):
        root = opt.dataroot
        phase = opt.phase
        pairLst = os.path.join(root, 'fasion-pairs-%s.csv' % phase)
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, '')
        bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' % phase)
        return image_dir, bonesLst, name_pairs

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs

    def obtain_bone(self,name,affine_matrix):
        string = self.annotation_file.loc[name]
        array = util_pose.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = util_pose.cords_to_map(array,self.load_size,self.opt.old_size,affine_matrix)
        pose = np.transpose(pose,(2,0,1))
        pose = torch.Tensor(pose)
        #pose = util_pose.kp2stick(array)
        #pose = np.transpose(pose,(2,0,1))
        #pose = torch.Tensor(pose)
        return pose

    def getRandomAffineParam(self):
        if self.opt.angle is not False:
            angle = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1])
        else:
            angle = 0
        if self.opt.scale is not False:
            scale   = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1])
        else:
            scale=1
        if self.opt.shift is not False:
            shift_x = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
            shift_y = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
        else:
            shift_x=0
            shift_y=0
        return angle, (shift_x,shift_y), scale
        
    def __len__(self):
        return self.dataset_size