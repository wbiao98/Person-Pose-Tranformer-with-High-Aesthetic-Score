import torch
import torch.nn as nn
import torchvision.models as models
from refinenet_4cascade import RefineNet4Cascade

class MyLoss(nn.Module):
    def __init__(self, w1=0.5, w2=0.5):
        super().__init__()

        self.w1 = w1
        self.w2 = w2
        self.BCELoss = nn.BCELoss(reduce=True, reduction='sum')
        self.MSELoss = nn.MSELoss()

    def forward(self, seg_out, depth_out, seg_target, depth_target):
        loss = self.w1 * self.BCELoss(seg_out, seg_target) + self.w2 * self.MSELoss(depth_out, depth_target)
        if loss > 10000:
            loss1 = self.BCELoss(seg_out, seg_target)
            loss2 = self.MSELoss(depth_out, depth_target)
        return loss

class ParsingLoss(nn.Module):
    def __init__(self):
        super(ParsingLoss, self).__init__()
        self.add_module('parsing', ParsingNet())
        self.loss = MyLoss()

    def __call__(self, x, y):
        x1_,x2_ = self.parsing(x)
        y1_,y2_ = self.parsing(y)
        lossxy = self.loss(x1_,x2_,y1_,y2_)
        return lossxy




class ParsingNet(nn.Module):
    def __init__(self):
        super(ParsingNet, self).__init__()
        net = RefineNet4Cascade(input_shape=(3, 256, 256), num_classes=40)
        net.load_state_dict(torch.load('RefineNet_1112_21_56_19.pkl'))
        net.eval()
        self.net = net

    def forward(self,x):
        return self.net(x)