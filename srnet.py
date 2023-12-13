import torch
import thop
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class SRNet(nn.Module):
    def __init__(self, conv=default_conv):
        super(SRNet, self).__init__()


        # define head module
        m_srnet = [conv(2, 64, 9),
        nn.ReLU(),
        conv(64, 32, 1),
        nn.ReLU(),
        conv(32, 2, 5),
        ]

        self.srnet = nn.Sequential(*m_srnet)

    def forward(self, x):
        out = self.srnet(x)

        return out


