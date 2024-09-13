import os
import torch
from .pix2pix_model import Pix2PixModel
from configs import CfgNode as CN


class AutoGANModel(Pix2PixModel):
    def __init__(self, opt):
        super(AutoGANModel, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = config
        _C.model.G.ckt_path = None
        _C.model.G.in_channel = 2
        _C.loaders.cat_inputs = True

        return _C

    def set_test_input(self, input, slice_idx, indices):
        self.real_A = (input.to(self.device) + 1) / 2
        self.slice_idx = slice_idx.to(self.device) if self.extra_b else None

    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = torch.clip(self.netG(self.real_A) * 2 - 1, -1, 1)   # G(A)
        if not self.isTrain:
            self.fake_B = self.fake_B.detach()
        if self.sg or (not self.isTrain and self.eval_seg):
            self.pred_mask = self.netSG(self.fake_B.unsqueeze(1)).squeeze(1)

    '''
    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pt' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                net.load_state_dict(state_dict.state_dict())
    '''
    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)
