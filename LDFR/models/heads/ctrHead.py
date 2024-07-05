import math

import torch
from torch import nn
import torch.nn.functional as F




def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class CtnetHead(nn.Module):
    def __init__(self, channels_in, train_cfg=None, test_cfg=None, down_ratio=4, final_kernel=1, head_conv=192,
                 branch_layers=0):
        super(CtnetHead, self).__init__()
        classes = channels_in//down_ratio
        self.head_conv = head_conv
        if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(channels_in, classes,
                              kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(classes, self.head_conv,kernel_size=final_kernel, stride=1,
                            padding=final_kernel // 2, bias=True))
            fc[-1].bias.data.fill_(-2.19)
            self.model = fc

    def forward(self, x):
        b,_,_,_=x.size()
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        z = self.model(x)
        return z.view(b,self.head_conv,-1)

    def init_weights(self):
        # ctnet_head will init weights during building
        pass
