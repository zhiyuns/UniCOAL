import torch
from collections import OrderedDict
from .base_model import BaseModel
from .utils import define_G, define_sg, get_sg_loss


class EDSRModel(BaseModel):
    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = config
        _C.model.G.input_nc = 1
        # of input image channels: 3 for RGB and 1 for grayscale
        _C.model.G.output_nc = 1
        _C.model.G.n_resblocks = 8
        _C.model.G.upsample = False
        _C.model.G.n_feats = 64
        _C.model.G.res_scale = 1
        # dropout for the generator
        _C.model.G.init_type = 'kaiming'
        # network initialization [normal | xavier | kaiming | orthogonal]

        return _C

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G',]
        # define networks (both generator and discriminator)
        self.netG = define_G(**opt.model.G, gpu_ids=self.gpu_ids)
        self.netG.requires_grad_(True)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.optimizer.lr_G, betas=(opt.optimizer.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        self.combine_ab = opt.model.combine_ab
        self.sg = opt.sg
        self.extra_b = opt.extra_b
        if self.extra_b:
            self.visual_names.append('extra_B')
        #     assert not self.combine_ab
        if self.sg or not self.isTrain:
            self.loss_names.append('seg')
            self.visual_names.append('pred_mask')
            self.criterionSeg = get_sg_loss(**opt.loss.SG)
        self.netSG = define_sg(**opt.model.SG, gpu_ids=self.gpu_ids)
        self.netSG.eval()
        self.netSG.requires_grad_(False)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.label = input[self.opt.loaders.label_internal_path[0]].to(self.device) if self.sg or not self.isTrain else None
        self.extra_B = input['B_extra'].to(self.device) if self.extra_b else None
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_test_input(self, input, slice_idx, indices):
        self.real_A = input.to(self.device)
        self.slice_idx = slice_idx.to(self.device) if self.extra_b else None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        if not self.isTrain:
            self.fake_B = self.fake_B.detach()
        if self.sg or not self.isTrain:
            self.pred_mask = self.netSG(self.fake_B)

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            if self.sg or not self.isTrain:
                self.pred_mask = torch.sigmoid(self.pred_mask)
            self.compute_visuals()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.loss.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1
        if self.sg:
            self.loss_seg = self.criterionSeg(self.pred_mask, self.label)
            self.loss_G += self.loss_seg
            self.pred_mask = torch.sigmoid(self.pred_mask)
        self.loss_G.backward()

    def optimize_parameters(self, **kwargs):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                img = getattr(self, name)
                img = img[:, 0, 0:1, :, :]
                visual_ret[name] = img
        return visual_ret
