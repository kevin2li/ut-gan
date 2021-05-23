import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from .srm_kernels import all_normalized_hpf_list

__all__ = ['ZhuNet']

# absult value operation
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, input):
        output = torch.abs(input)
        return output

# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output

# Pre-processing Module
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)

        return output

# https://gist.github.com/erogol/a324cc054a3cdc30e278461da9f1a05e
class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)

            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

class ZhuNet(nn.Module):
    def __init__(self):
        super(ZhuNet, self).__init__()
        self.layer1 = HPF()
        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1, groups=30),
            ABS(),
            nn.Conv2d(60, 30, kernel_size=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1, groups=30),
            nn.Conv2d(60, 30, kernel_size=1),
            nn.BatchNorm2d(30)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer8 = SPPLayer(3)
        self.fc = nn.Sequential(
            nn.Linear(128*21, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        
        # sepconv
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3 + out1

        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.fc(out8)

        return out9