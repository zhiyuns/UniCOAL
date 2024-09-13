import torch
from collections import OrderedDict
from .pix2pix_model import Pix2PixModel
from configs import CfgNode as CN


class ResViTModel(Pix2PixModel):
    def __init__(self, opt):
        super(ResViTModel, self).__init__(opt)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.optimizer.lr_G, betas=(opt.optimizer.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.optimizer.lr_D, betas=(opt.optimizer.beta1, 0.999))
        self.netG.requires_grad_(True)
        self.netD.requires_grad_(True)

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = config
        _C.loaders.slice_num = 4
        _C.model.G.input_dim = 1
        # of input image channels: 3 for RGB and 1 for grayscale
        _C.model.G.output_dim = 1
        # should be removed
        _C.model.combine_input = True
        _C.model.G.vis = False
        _C.model.G.img_size = 256
        _C.model.G.img_size = 256
        _C.model.G.pre_trained_resnet = None
        _C.model.G.pre_trained_trans = None

        _C.model.G.encoder_kwargs = CN()
        _C.model.G.encoder_kwargs.hidden_size = 768
        _C.model.G.encoder_kwargs.num_layers = 12
        _C.model.G.encoder_kwargs.block_kwargs = CN()
        _C.model.G.encoder_kwargs.block_kwargs.hidden_size = 768
        _C.model.G.encoder_kwargs.block_kwargs.mlp_kwargs = CN()
        _C.model.G.encoder_kwargs.block_kwargs.mlp_kwargs.mlp_dim = 3072
        _C.model.G.encoder_kwargs.block_kwargs.mlp_kwargs.dropout_rate = 0.1
        _C.model.G.encoder_kwargs.block_kwargs.attention_kwargs = CN()
        _C.model.G.encoder_kwargs.block_kwargs.attention_kwargs.num_heads = 12
        _C.model.G.encoder_kwargs.block_kwargs.attention_kwargs.hidden_size = 768
        _C.model.G.encoder_kwargs.block_kwargs.attention_kwargs.dropout_rate = 0.0

        _C.model.G.block_kwargs = CN()
        _C.model.G.block_kwargs.hidden_size = 768
        _C.model.G.block_kwargs.embedding_kwargs = CN()
        _C.model.G.block_kwargs.embedding_kwargs.grid = [16, 16]
        _C.model.G.block_kwargs.embedding_kwargs.dropout_rate = 0.1

        _C.model.D.input_nc = 2
        # of input image channels: 6 for RGB and 2 for grayscale
        _C.model.D.ndf = 64
        # of gen filters in the last conv layer
        _C.model.D.norm = 'batch'
        # instance normalization or batch normalization [instance | batch | none]
        _C.model.D.n_layers = 3
        # only used if netD==n_layers
        _C.model.D.init_type = 'normal'
        # network initialization [normal | xavier | kaiming | orthogonal]
        _C.model.D.init_gain = 0.02
        # scaling factor for normal, xavier and orthogonal

        return _C

    def set_input(self, input):
        super(ResViTModel, self).set_input(input)
        if self.opt.model.combine_input:
            if self.opt.loaders.random_modality_in:
                class_in = input['A_class'].to(self.device)
                class_in = torch.argmax(class_in, dim=1)
                real_A = torch.zeros(self.real_A.size()[0], len(self.opt.loaders.raw_internal_path_in), self.real_A.size()[2], self.real_A.size()[3]).to(self.device)
                real_A[:, class_in, ...] = self.real_A.squeeze(1)
                self.real_A = real_A
            else:
                # exclude the specific class
                class_in = input['B_class'].to(self.device)
                class_in = torch.argmax(class_in, dim=1)
                b, c, h, w = self.real_A.size()
                real_A = torch.zeros(b, len(self.opt.loaders.raw_internal_path_in), h, w).to(self.device)
                b, c, h, w = real_A.size()
                # Get the shape of A and B
                empty = torch.zeros(b, 1, h, w).to(self.device)
                # Initialize the output tensor with the same shape as A
                output = torch.empty_like(real_A)
                # Iterate over each batch element
                for i in range(b):
                    # Create an index tensor for the current batch element
                    index_tensor = torch.arange(c).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)
                    index_tensor = (index_tensor == class_in[i]).type(torch.bool)
                    # Concatenate B into A along the channel dimension for the current batch element
                    A_before_insert = self.real_A[i, :class_in[i], :, :]
                    A_after_insert = self.real_A[i, class_in[i]:, :, :]
                    output[i] = torch.cat([A_before_insert, empty[i], A_after_insert], dim=0)
                self.real_A = output
                
        if len(self.opt.loaders.raw_internal_path_in) > 1:
            self.real_B = input['B_all'].to(self.device)

    def set_test_input(self, input, slice_idx, indices):
        super(ResViTModel, self).set_test_input(input, slice_idx, indices)
        if self.opt.model.combine_input:
            if self.opt.loaders.random_modality_in:
                class_in = torch.zeros([self.real_A.size()[0], len(self.opt.loaders.raw_internal_path_out)]).pin_memory().to(self.device)
                class_in[:, self.opt.loaders.test.original_modality] = 1
                class_in = torch.argmax(class_in, dim=1)
                real_A = torch.zeros(self.real_A.size()[0], len(self.opt.loaders.raw_internal_path_in), self.real_A.size()[2], self.real_A.size()[3]).to(self.device)
                real_A[:, class_in, ...] = self.real_A.squeeze(1)
                self.real_A = real_A
            else:
                class_in = torch.zeros([self.real_A.size()[0], len(self.opt.loaders.raw_internal_path_out)]).pin_memory().to(self.device)
                class_in[:, self.opt.loaders.test.target_modality] = 1
                class_in = torch.argmax(class_in, dim=1)
                real_A = torch.zeros(self.real_A.size()[0], len(self.opt.loaders.raw_internal_path_in), self.real_A.size()[2], self.real_A.size()[3]).to(self.device)
                # Get the shape of A and B
                b, c, h, w = real_A.size()
                empty = torch.zeros(self.real_A.size()[0], 1, self.real_A.size()[2], self.real_A.size()[3]).to(self.device)
                # Initialize the output tensor with the same shape as A
                output = torch.empty_like(real_A)
                # Iterate over each batch element
                for i in range(b):
                    # Create an index tensor for the current batch element
                    index_tensor = torch.arange(c).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)
                    index_tensor = (index_tensor == class_in[i]).type(torch.bool)
                    # Concatenate B into A along the channel dimension for the current batch element
                    A_before_insert = self.real_A[i, :class_in[i], :, :]
                    A_after_insert = self.real_A[i, class_in[i]:, :, :]
                    output[i] = torch.cat([A_before_insert, empty[i], A_after_insert], dim=0)
                self.real_A = output

    def test(self):
        super(ResViTModel, self).test()
        if len(self.opt.loaders.raw_internal_path_in) > 1:
            # class_out = torch.zeros([self.real_A.size[0], len(self.opt.loaders.raw_internal_path_out)]).pin_memory().to(self.device)
            # class_out[:, self.opt.loaders.test.target_modality] = 1
            if hasattr(self, 'real_B'):
                self.real_B = self.real_B[:, self.opt.loaders.test.target_modality, ...].unsqueeze(1) 
            self.fake_B = self.fake_B[:, self.opt.loaders.test.target_modality, ...].unsqueeze(1)

    def run_G(self, cond_img, noise_mode='random'):
        img = self.netG(cond_img)
        return img

    def run_D(self, img, **kwargs):
        return self.netD(img)

    def forward_ema(self):
        ref_img = self.extra_B if self.extra_b else self.real_B
        self.fake_B = self.netG(self.real_A)  # G(A)
        if self.sg or not self.isTrain:
            self.fake_B = self.fake_B.detach()
            with torch.no_grad():
                self.pred_mask = self.netSG(self.fake_B.unsqueeze(1))
                self.pred_mask = torch.sigmoid(self.pred_mask.squeeze(1))

    def optimize_parameters(self, **kwargs):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                img = getattr(self, name)
                img = img[:, 0:1, :, :]
                visual_ret[name] = img
        return visual_ret
