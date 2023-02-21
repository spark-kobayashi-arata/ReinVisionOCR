from collections import namedtuple
import torch.nn as nn
from torch import Tensor
from torchvision.models import vgg
from torchvision.models.vgg import VGG16_BN_Weights


__all__ = [
    "vgg16_bn",
    "VGG16_BN_Weights",
    "init_weights",
]


def init_weights(modules) -> None:
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class vgg16_bn(nn.Module):
    def __init__(
        self,
        weights:VGG16_BN_Weights=VGG16_BN_Weights.IMAGENET1K_V1,
        freeze:bool=True,
    ):
        super().__init__()
        
        vgg_pretrained_features = vgg.vgg16_bn(weights=weights).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        # conv2_2
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # conv3_3
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # conv4_3
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # # conv5_3
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        # fc6, fc7 without atrous conv
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )
        
        if weights is not VGG16_BN_Weights.IMAGENET1K_V1:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
        
        # no pretrained model for fc6 and fc7
        init_weights(self.slice5.modules())
        
        if freeze:
            for param in self.slice1.parameters():
                # only first conv
                param.requires_grad = False
    
    def forward(self, x:Tensor) -> Tensor:
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
