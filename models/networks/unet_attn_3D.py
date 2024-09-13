import torch
import torch.nn as nn
import functools

#Sapce Based Attention 3D
class SBA_Block_3D(nn.Module):
    def __init__(self, in_channel, r):
        super(SBA_Block_3D, self).__init__()
        
        self.query_conv = nn.Conv3d(in_channels = in_channel , out_channels = int(in_channel/r) , kernel_size= 1)
        self.key_conv = nn.Conv3d(in_channels = in_channel , out_channels = int(in_channel/r) , kernel_size= 1)
        self.value_conv = nn.Conv3d(in_channels = in_channel , out_channels = in_channel , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize,C, depth, width ,height = x.size()
        
        out_q = self.query_conv(x).view(m_batchsize,-1,depth*width*height).permute(0,2,1) # B X CX(N)
        out_k = self.key_conv(x).view(m_batchsize,-1,depth*width*height) # B X C x (*W*H)
        energy =  torch.bmm(out_q,out_k) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        # attention = energy # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,depth*width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,depth,width,height)
        
        out = self.gamma*out + x
        return out

class UnetSkipConnectionBlock_3D(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, has_att=False):
        super(UnetSkipConnectionBlock_3D, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if has_att:
                att1=SBA_Block_3D(input_nc, 8)
                att2=SBA_Block_3D(outer_nc, 8)

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            elif has_att:
                model = [att1] + down + [submodule] + up + [att2]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator_withatt_3D(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_withatt_3D, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock_3D(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock_3D(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, has_att=True)
        unet_block = UnetSkipConnectionBlock_3D(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3D(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3D(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)