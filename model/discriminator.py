import torch
from torch import nn
from model.modules import ResBlockEncoder
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import functools

class ResDiscriminator(nn.Module):
    def __init__(self,input_nc=3, ndf=32, img_f=128, layers=4):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        nonlinearity = nn.LeakyReLU(0.1)
        self.nonlinearity = nonlinearity
        self.block0 = ResBlockEncoder(input_nc, ndf, ndf,nonlinearity)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf)
            block = ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev,nonlinearity)
            setattr(self, 'encoder' + str(i), block)

        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 1))


    def forward(self,x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out