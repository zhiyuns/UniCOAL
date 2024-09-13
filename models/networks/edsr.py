import torch.nn as nn
import torch


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSR(nn.Module):
    def __init__(self, input_nc, output_nc, n_feats, n_resblocks, upsample, res_scale, up_scale=None,
                 conv=default_conv, kernel_size=3, act=nn.ReLU(True)):
        super(EDSR, self).__init__()
        self.up_scale = up_scale
        self.upsample = upsample
        # define head module
        m_head = [conv(input_nc, n_feats, kernel_size)]
        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
                  for _ in range(n_resblocks)]
        m_tail = [conv(n_feats, output_nc, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):  # x: b,c,h,w,d

        w, h, d = x.shape[2:]
        res = x + self.body(self.head(x))
        if self.upsample:
            res = torch.nn.functional.interpolate(res, size=(
                self.up_scale[0] * (w - 1) + 1, self.up_scale[1] * (h - 1) + 1, self.up_scale[2] * (d - 1) + 1),
                                                  mode='trilinear', align_corners=True)
        res = self.tail(res)
        return res
