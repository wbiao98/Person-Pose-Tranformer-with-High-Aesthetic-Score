import numpy as np
import torch
import os
from TransPoseNet5 import TransposeNet
import glob
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from options.test_options import TestOptions
from PoseDataSet import Pose_dataLoder
from torchvision.transforms import ToPILImage
from util import util_pose



def test_net(net,opt,device,batch_size=1):
    test_dataset = Pose_dataLoder()
    test_dataset.initialize(opt)
    train_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    net.eval()
    for i, input in enumerate(train_loader):
        net.set_input(input)
        out = net.test()
        out_img = out.cpu().detach()
        print("process image :",i)
        name1 = input['P1_path'][0].strip('.jpg')+'_2_'+input['P2_path'][0]
        name2 = input['P1_path'][0].strip('.jpg')+'_condition.jpg'
        name3 = input['P2_path'][0].strip('.jpg')+'_traget.jpg'
        path1 ='./result/'+name1
        path2 = './result/' + name2
        path3 = './result/' + name3
        out_img1 = util_pose.tesor2im(out_img)
        out_img2 = util_pose.tesor2im(input['P1'])
        out_img3 = util_pose.tesor2im(input['P2'])
        result = torch.cat([input['P1'], out_img, input['P2']], 3)
        out_img1 = util_pose.tesor2im(result)
        util_pose.save_image(out_img1, path1)
        util_pose.save_image(out_img2, path2)
        util_pose.save_image(out_img3, path3)
        # img = ToPILImage(out_img)
        # img.save(path)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    opt = TestOptions().parse(
    net = TransposeNet(opt,in_channels=21,out_channels=3,device=device)
    net.load_state_dict(torch.load('pptnet_7_dict.pth', map_location=device))
    net.to(device=device)
    test_net(net,opt,device)


