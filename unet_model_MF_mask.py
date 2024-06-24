# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import sys
#sys.path.append('/content/drive/MyDrive/Medical Imaging & Big Data/Progetto')
sys.path.append('/content/drive/MyDrive/Progetto')

import torch
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(160, 320)
        self.down3 = down(320, 640)
        self.down4 = down(640, 640)
        self.up1 = up(1280, 320)
        self.up2 = up(640, 160)
        self.up3 = up(320, 32)
        self.up4 = up(112, 16)
        self.outc = outconv(16, n_classes)

    def forward(self, banda1, banda2, banda3, banda4, banda5):
        x11 = self.inc(banda1)
        x12 = self.inc(banda2)
        x13 = self.inc(banda3)
        x14 = self.inc(banda4)
        x15 = self.inc(banda5)
        x1 = torch.cat([x11, x12, x13, x14, x15], dim=1)
        x21 = self.down1(x11)
        x22 = self.down1(x12)
        x23 = self.down1(x13)
        x24 = self.down1(x14)
        x25 = self.down1(x15)
        x2 = torch.cat([x21, x22, x23, x24, x25], dim=1)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)