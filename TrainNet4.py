from TransPoseNet5 import TransposeNet
from PoseDataSet import Pose_dataLoder
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import optim
from options.pose_options import PoseOptions

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = PoseOptions().parse()
    train_dataset = Pose_dataLoder()
    train_dataset.initialize(opt)
    train_loader = DataLoader(dataset=train_dataset,batch_size=4,shuffle=True)
    net = TransposeNet(opt,in_channels=21,out_channels=3,device=device)
    for epoch in range(20):
        for i,input in enumerate(train_loader):
            net.set_input(input)
            net.optimize_parameters()
        opt_dict = net.get_optimize()
        torch.save(net, 'pptnet_train.pth')
        torch.save(net.state_dict(), 'pptnet_train_dict.pth')



