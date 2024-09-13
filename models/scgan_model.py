import torch
import random
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel

from .utils import define_G, define_D, GANLoss, ImagePool

from torchvision import models

class SCGANModel(BaseModel):
    
    @staticmethod
    def modify_commandline_options(config, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            config          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.

        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        """
        _C = config
        _C.loaders.thickness_post = []
        _C.model.G.input_nc = 1
        # of input image channels: 3 for RGB and 1 for grayscale
        _C.model.G.output_nc = 1
        _C.model.G.ngf = 64
        # of gen filters in the last conv layer
        _C.model.G.norm = 'batch'
        # instance normalization or batch normalization [instance | batch | none]
        _C.model.G.dropout = True
        # dropout for the generator
        _C.model.G.init_type = 'normal'
        # network initialization [normal | xavier | kaiming | orthogonal]
        _C.model.G.init_gain = 0.02
        # scaling factor for normal, xavier and orthogonal

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
    
    def name(self):
        return 'Pix2PixModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # load/define networks
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt.down_samp)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        self.netG = define_G(**opt.model.G, gpu_ids=self.gpu_ids)
        self.model_names = ['G']
        #self.vgg=VGG16().cuda()
        if self.isTrain:
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD = define_D(**opt.model.D, gpu_ids=self.gpu_ids)
            self.model_names.append('D')
            self.fake_AB_pool = ImagePool(pool_size=0)
            # define loss functions
            self.criterionGAN = GANLoss(opt.loss.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            # self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.optimizer.lr_G, betas=(opt.optimizer.beta1, 0.999), weight_decay=0.001)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.optimizer.lr_D, betas=(opt.optimizer.beta1, 0.999), weight_decay=0.001)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.netG.requires_grad_(True)
            self.netD.requires_grad_(True)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B'].to(self.device)
        self.input_B = input['B' if AtoB else 'A'].to(self.device)
        if len(self.opt.loaders.thickness_post) > 0: 
            input_A = []
            for idx in range(self.input_A.size()[0]):
                each_A = self.input_A[idx:idx+1, ...]
                thickness = random.choice(self.opt.loaders.thickness_post)
                if thickness > 1:
                    each_A = each_A[:, :, ::thickness, :, :]
                    each_A = torch.nn.functional.interpolate(each_A, size=(self.input_A.shape[2], self.input_A.shape[3], self.input_A.shape[4]), mode='trilinear', align_corners=True)
                input_A.append(each_A)
            self.input_A = torch.cat(input_A, dim=0)

    def set_test_input(self, input, slice_idx, indices):
        self.input_A = input.to(self.device)
        if len(self.opt.loaders.thickness_post) > 0: 
            input_A = []
            for idx in range(self.input_A.size()[0]):
                each_A = self.input_A[idx:idx+1, ...]
                thickness = random.choice(self.opt.loaders.thickness_post)
                if thickness > 1:
                    each_A = each_A[:, :, ::thickness, :, :]
                    each_A = torch.nn.functional.interpolate(each_A, size=(self.input_A.shape[2], self.input_A.shape[3], self.input_A.shape[4]), mode='trilinear', align_corners=True)
                input_A.append(each_A)
            self.input_A = torch.cat(input_A, dim=0)
    
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A, volatile=True)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B, volatile=True) if hasattr(self, 'input_B') else None

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.loss.lambda_L1
        
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 
        
        self.loss_G.backward()

    def optimize_parameters(self, **kwargs):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                img = getattr(self, name)
                img = img[:, 0, 0:1, :, :]
                visual_ret[name] = img
        return visual_ret

#VGG        
class VGG_OUTPUT(object):

    def __init__(self,relu2_2):
        self.__dict__ = locals()


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
#        for x in range(9, 16):
#            self.slice3.add_module(str(x), vgg_pretrained_features[x])
#        for x in range(16, 23):
#            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h = self.slice1(X)
        #h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
#        h = self.slice3(h)
#        h_relu3_3 = h
#        h = self.slice4(h)
#        h_relu4_3 = h
        return VGG_OUTPUT(h_relu2_2)


