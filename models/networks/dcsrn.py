import torch.nn as nn
import torch


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class DenseBlock(nn.Module):
    def __init__(self, conv, in_features, out_features, kernel_size, bias=True, bn=False, act=nn.ELU()):

        super(DenseBlock, self).__init__()
        m = []
        if bn:
            m.append(nn.BatchNorm3d(in_features))
        m.append(act)
        m.append(conv(in_features, out_features, kernel_size, bias=bias))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res


class DCSRN(nn.Module):
    def __init__(self, input_nc, output_nc, layers, n_feats, kernel_size=3, conv=default_conv):
        super(DCSRN, self).__init__()
        self.layers = layers
        self.stem = conv(input_nc, n_feats * 2, kernel_size)
        
        for layer in range(layers):
            in_features = (2 + layer) * n_feats
            out_features = n_feats
            name = f'L{layer}'
            layer = DenseBlock(conv, in_features, out_features, kernel_size, bn=True)
            setattr(self, name, layer)

        in_features = (2 + layers) * n_feats
        m_tail = [conv(in_features, output_nc, kernel_size)]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):  # x: b,c,h,w,d

        x1 = self.stem(x)
        all_x = [x1]
        for layer in range(self.layers):
            name = f'L{layer}'
            layer = getattr(self, name)
            all_x.append(layer(torch.cat(all_x, 1)))

        x1 = self.tail(torch.cat(all_x, 1))
        x = x + x1

        return x
