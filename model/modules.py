import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

class Space2Depth(nn.Module):
    def __init__(self,block_size):
        super.__init__()
        self.bs = block_size

    def forward(self,x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)
        return x

class Depth2Space(nn.Module):
    def __init__(self,block_size):
        super.__init__()
        self.bs = block_size

    def forward(self,x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)
        return x

class IDAct(nn.Module):
    def forward(self,input):
        return  input

class NormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super.__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self,x):
        out = self.conv(x)
        out = self.gamma*out + self.beta
        return out

class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d):
        super.__init__()
        if out_channels == None:
            self.down = conv_layer(
                channels, channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.down = conv_layer(
                channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self,x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self,in_channels, out_channels, subpixel=True, conv_layer=NormConv2d):
        super.__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1)
            self.op2 = DepthToSpace(block_size=2)
        else:
            # channels have to be bisected because of formely concatenated skips connections
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.op2 = IDAct()

    def forward(self,x):
        out = self.up(x)
        out = self.op2(out)
        return out

class VUnetResnetBlock(nn.Module):
    def __init__(self, out_channels,use_skip=False,kerner_size=3,activate=False,conv_layer=NormConv2d,gated=False,final_act=False,dropout_prob=0.0):
        super.__init__()
        self.dout = nn.Dropout(p=dropout_prob)
        self.use_skip = use_skip
        self.gated = gated
        if self.use_skip:
            self.conv2d = conv_layer(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.pre = conv_layer(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1,
            )
        else:
            self.conv2d = conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        if self.gated:
            self.conv2d2 = conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.dout2 = nn.Dropout(p=dropout_prob)
            self.sigm = nn.Sigmoid()
        if activate:
            self.act_fn = nn.LeakyReLU() if final_act else nn.ELU()
        else:
            self.act_fn = IDAct()

    def forward(self,x,a=None):
        x_prc = x

        if self.use_skip:
            assert a is not None
            a = self.pre(a)
            x_prc = torch.cat([x_prc, a], dim=1)

        x_prc = self.act_fn(x_prc)
        x_prc = self.dout(x_prc)
        x_prc = self.conv2d(x_prc)

        if self.gated:
            x_prc = self.act_fn(x_prc)
            x_prc = self.dout(x_prc)
            x_prc = self.conv2d2(x_prc)
            a, b = torch.split(x_prc, 2, 1)
            x_prc = a * self.sigm(b)

        return x + x_prc

class ChannelAttention(nn.Module):
    def __init__(self,in_planes,ration=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes,in_planes//16,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16,in_planes,1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention,self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class DownBasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownBasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = self.ca(out) * out
        out = self.sa(out) * out

        out = self.relu(out)

        return out

class UpBasicBlock(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(UpBasicBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class UpCBAMBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpCBAMBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = self.ca(out) * out
        out = self.sa(out) * out

        out = self.relu(out)

        return out

class Inconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Inconv,self).__init__()
        self.conv= DownBasicBlock(in_channels,out_channels)

    def forward(self,x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Down, self).__init__()
        self.mconv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=4, stride=2, padding=1),
            DownBasicBlock(in_channels,out_channels)
        )

    def forward(self,x):
        x = self.mconv(x)
        return x

class Up(nn.Module):
    def __init__(self,in_channels,out_channels,Transpose=False):
        super(Up, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,2,stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels,in_channels//2,kernel_size=2,padding=0),
                nn.ReLU(inplace=True)
            )

        self.conv = UpBasicBlock(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2,x1],dim=1)
        x = self.conv(x)
        return x
class Upcbam(nn.Module):
    def __init__(self,in_channels,out_channels,Transpose=False):
        super(Upcbam, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,2,stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels,in_channels//2,kernel_size=2,padding=0),
                nn.ReLU(inplace=True)
            )

        self.conv = UpCBAMBlock(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2,x1],dim=1)
        x = self.conv(x)
        return x

class LastLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(LastLayer,self).__init__()
        self.conv = UpBasicBlock(in_channels,out_channels)

    def forward(self,x1,x2):
        
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
        return self.conv(x)

class OutConv2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv2,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.conv(un_pool(x,2))

class ResBlockEncoder(nn.Module):
    def __init__(self,input_nc, output_nc, hidden_nc, nonlinearity= nn.LeakyReLU()):
        super(ResBlockEncoder, self).__init__()
        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        conv1 = SpectralNorm(nn.Conv2d(input_nc,hidden_nc,kernel_size=3,stride=1,padding=1))
        conv2 = SpectralNorm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1))
        bypass = SpectralNorm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0))
        self.model = nn.Sequential(nonlinearity, conv1,nonlinearity, conv2)
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)


    def forward(self,x):
        out1 = self.model(x) 
        out2 = self.shortcut(x)
        out = out1+out2
        return out

def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True
                
class PoseEncoder(nn.Module):
    def __init__(self,in_channels):
        super(PoseEncoder, self).__init__()
        self.inconv = Inconv(in_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)

    def forward(self,x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1,x2,x3,x4

class PoseDecoder(nn.Module):
    def __init__(self,in_channels):
        super(PoseDecoder, self).__init__()
        self.inconv = LastLayer(in_channels,512)
        self.up1 = Up(512,256)
        self.up2 = Up(256,128)
        self.up3 = Up(128,64)
        self.out = OutConv(64,3)

    def forward(self,x,y,x1,x2,x3):
        out = self.inconv(x,y)
        out = self.up1(out,x3)
        out = self.up2(out,x2)
        out = self.up3(out,x1)
        out = self.out(out)
        return out



def un_pool(input,scale):
    return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)

class ResidualConvUnit(nn.Module):
    def __init__(self,features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super(MultiResolutionFusion, self).__init__()

        _, max_h, max_w = max(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, h, w = shape
            if max_h % h != 0:
                raise ValueError("max_size not divisble by shape {}".format(i))

            self.scale_factors.append(max_h // h)
            self.add_module(
                "resolve{}".format(i),
                nn.Conv2d(
                    feat,
                    out_feats,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))
        self.ca = ChannelAttention(out_feats)
        self.sa = SpatialAttention()
            

    def forward(self, *xs):   
        #print(self.scale_factors)
        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = un_pool(output, self.scale_factors[0])
            #print("output",output.shape)

        for i, x in enumerate(xs[1:], 1):
            tmp_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                tmp_out = un_pool(tmp_out, self.scale_factors[i])
            output = output + tmp_out
        output = self.ca(output)*output
        output = self.sa(output)*output
        return output


class ChainedResidualPool(nn.Module):
    def __init__(self, feats, block_count=4):
        super(ChainedResidualPool, self).__init__()

        self.block_count = block_count
        self.relu = nn.ReLU(inplace=False)
        for i in range(0, block_count):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(0, self.block_count):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super(BaseRefineNetBlock, self).__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats), residual_conv_unit(feats)))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)