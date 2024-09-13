import os
import torch
from .pix2pix_model import Pix2PixModel
from configs import CfgNode as CN
from collections import OrderedDict
from torch.autograd import Variable

class ProvoGANModel(Pix2PixModel):
    def __init__(self, opt):
        super(ProvoGANModel, self).__init__(opt)
        self.netG.requires_grad_(True)
        self.netD.requires_grad_(True)

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        super(ProvoGANModel, ProvoGANModel).modify_commandline_options(config, is_train)
        _C = config
        _C.loaders.train.ori_file_path = ''
        _C.loaders.provo_stage = 1
        return _C

    
    def set_input(self, input):
        super(ProvoGANModel, self).set_input(input)
        if 'A_fake' in input:
            self.fake_A = Variable(input['A_fake'].to(self.device))
        else:
            self.fake_A = None

    def set_test_input(self, input, slice_idx, indices):
        self.real_A = input.to(self.device)
        self.fake_A = slice_idx.to(self.device)
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.fake_A is not None:
            self.residual = self.netG(self.real_A)   # G(A)
            self.fake_B = self.fake_A-2*self.residual
        else:
            self.fake_B = self.netG(self.real_A)
        if not self.isTrain:
            self.fake_B = self.fake_B.detach()
        if self.sg or (not self.isTrain and self.eval_seg):
            self.pred_mask = self.netSG(self.fake_B.unsqueeze(1)).squeeze(1)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                img = getattr(self, name)
                img = img[:, 0:1, :, :]
                visual_ret[name] = img
        return visual_ret
