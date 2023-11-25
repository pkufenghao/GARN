import torch
import torch.nn.functional as F
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class GARN_Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, kernel = 3):

        m = []
        m.append(nn.ConvTranspose2d(n_feats, n_feats, kernel_size=(kernel, scale), stride=(1, scale), padding=(1, 0)))
        m.append(nn.BatchNorm2d(n_feats))

        super(GARN_Upsampler, self).__init__(*m)

class GARB(nn.Module):
    def __init__(self, n_feats, n_factors):
        super(GARB, self).__init__()

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)

        # layers
        self.fc1 = nn.Conv2d(n_feats, n_feats//n_factors, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Conv2d(n_feats//n_factors, n_feats, kernel_size=1)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        # Squeeze
        w = F.avg_pool2d(out, kernel_size=(out.size(2),out.size(3)))
        w = self.fc1(w)
        w = self.relu2(w)
        w = self.fc2(w)
        w = torch.sigmoid(w)

        # Excitation
        out = out * w

        out = out + x
        return out 

class GARN(nn.Module):
    def __init__(self, n_scale = 2, num_blocks = 4, n_feats = 32, n_factors = 4, conv=default_conv):
        super(GARN, self).__init__()

        kernel_size = 3

        # define head module
        m_head = [conv(2, n_feats, kernel_size)]

        # define body module
        m_body = [GARB(n_feats, n_factors) for _ in range(num_blocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            GARN_Upsampler(n_scale, n_feats, kernel_size),
            conv(n_feats, 2, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

