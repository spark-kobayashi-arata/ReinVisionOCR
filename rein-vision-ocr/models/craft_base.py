import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

from .vgg16_bn import *


__all__ = [
    "CRAFTBase",
]


class double_conv(nn.Module):
    def __init__(
        self,
        in_channels:int,
        mid_channels:int,
        out_channels:int,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:Tensor):
        x = self.conv(x)
        return x


class CRAFTBase(nn.Module):
    MEAN_NORM = (0.485, 0.456, 0.406)
    STD_NORM = (0.229, 0.224, 0.225)
    
    def __init__(
        self,
        pretrained:bool=False,
        freeze:bool=False,
    ):
        super().__init__()
        
        self.is_cuda = torch.cuda.is_available()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN_NORM, std=self.STD_NORM),
        ])
        
        if pretrained:
            weights = VGG16_BN_Weights.IMAGENET1K_V1
        else:
            weights = VGG16_BN_Weights.DEFAULT
        
        # Base network
        self.basenet = vgg16_bn(weights, freeze)
        
        # U network
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)
        
        num_class = 1
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )
        
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
    
    def forward(self, x:Tensor) -> Tensor:
        # Base network
        sources = self.basenet(x)
        
        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        
        y = F.interpolate(y, size=sources[2].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        
        y = F.interpolate(y, size=sources[3].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        
        y = F.interpolate(y, size=sources[4].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        y = self.upconv4(y)
        
        y = self.conv_cls(y)
        
        return y
    
    @classmethod
    def load_from_pretrained(cls, path:str):
        """学習済みモデルからCRAFTをセットアップ

        Args:
            path (str): _description_

        Returns:
            _type_: _description_
        """
        checkpoint = torch.load(path)
        
        model = cls(**checkpoint["hyper_parameters"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        if model.is_cuda:
            model.cuda()
        
        return model
