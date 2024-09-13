import torch
import numpy as np
from .comodgan_model import CoModGANModel, CN
from models.networks.UniCOAL.torch_utils.ops import upfirdn2d
from .networks.CoModGAN.torch_utils import misc
from torchvision import models
from segment_anything import SamPredictor, sam_model_registry

class UniCOALModel(CoModGANModel):
    def __init__(self, opt):
        super(UniCOALModel, self).__init__(opt)
        self.modality_list = opt.loaders.raw_internal_path_out
        self.cross_section_consistency = opt.loss.lambda_consistency > 0
        self.blur_sigma = 0
        self.thickness = opt.loaders.thickness[0]
        if self.cross_section_consistency:
            self.visual_names.append('fake_B_ema')

        if opt.loss.lambda_vgg > 0:
            self.vgg = VGG16().to(self.device)
            self.loss_names.append('VGG')

        if opt.loss.lambda_sam > 0:
            self.sam = sam_model_registry["vit_b"](checkpoint="../segment-anything/checkpoints/sam_vit_b_01ec64.pth").to(self.device)
            for param in self.sam.parameters():
                param.requires_grad = False
            self.loss_names.append('SAM')

    def run_G(self, cond_img, gen_index, update_emas=False, noise_mode='random'):
        ref_img = self.extra_B if self.extra_b else self.real_B
        ws = self.G_mapping(z=self.gen_z, c=self.gen_c, index=gen_index, img_in=ref_img, update_emas=False)
        if self.style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                 torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.G_mapping(z=torch.randn_like(self.gen_z), c=self.gen_c, img_in=ref_img, skip_w_avg_update=True)[:, cutoff:]
        img = self.G_synthesis(ws, c=self.gen_c, index=gen_index, img_in=cond_img, update_emas=False, noise_mode=noise_mode)
        return img

    def run_G_emas(self, cond_img, gen_index):
        ref_img = self.extra_B if self.extra_b else self.real_B
        self.fake_B_ema = self.netG_ema(z=self.gen_z, c=self.gen_c, index=gen_index, cond_img=cond_img, ref_img=ref_img, noise_mode='const')
        return self.fake_B_ema

    def run_D(self, img, **kwargs):
        blur_size = np.floor(self.blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(self.blur_sigma).square().neg().exp2()
            img = upfirdn2d.filter2d(img, f / f.sum())

        return self.netD(img, **kwargs)

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = config
        _C.loaders.slice_num = 3
        _C.loaders.input_thickness_match = False

        _C.model.G.pre_trained_resnet = None
        _C.model.G.z_dim = 512
        _C.model.G.w_dim = 512
        _C.model.G.c_dim = 0
        _C.model.G.index_dim = 0
        _C.model.G.img_resolution = 256
        _C.model.G.img_channels_in = 1
        _C.model.G.img_channels_out = 1

        _C.model.G.synthesis_kwargs = CN()
        _C.model.G.synthesis_kwargs.channel_base = int(0.5 * 32768)
        _C.model.G.synthesis_kwargs.channel_max = 512
        _C.model.G.synthesis_kwargs.num_layers = 14
        _C.model.G.synthesis_kwargs.num_critical = 2
        _C.model.G.synthesis_kwargs.first_cutoff = 2
        _C.model.G.synthesis_kwargs.first_stopband = 2**2.1
        _C.model.G.synthesis_kwargs.last_stopband_rel = 2**0.3
        _C.model.G.synthesis_kwargs.margin_size = 10
        _C.model.G.synthesis_kwargs.output_scale = 0.25
        _C.model.G.synthesis_kwargs.skip_resolution = 128
        _C.model.G.synthesis_kwargs.freeze_layer = -1
        _C.model.G.synthesis_kwargs.coord_conv = False
        # layer kwargs
        _C.model.G.synthesis_kwargs.conv_kernel = 3
        _C.model.G.synthesis_kwargs.filter_size = 6
        _C.model.G.synthesis_kwargs.lrelu_upsampling = 2
        _C.model.G.synthesis_kwargs.use_radial_filters = False
        _C.model.G.synthesis_kwargs.conv_clamp = 256
        _C.model.G.synthesis_kwargs.magnitude_ema_beta = 0.5 ** (16 / (20 * 1e3))  # depend on bs
        _C.model.G.synthesis_kwargs.cond_mod = True

        _C.model.G.synthesis_kwargs.encoder_kwargs = CN()
        _C.model.G.synthesis_kwargs.encoder_kwargs.name = 'ReversedEncoder'
        _C.model.G.synthesis_kwargs.encoder_kwargs.dropout_rate = 0.5
        _C.model.G.synthesis_kwargs.encoder_kwargs.modulated = False
        _C.model.G.synthesis_kwargs.encoder_kwargs.mapping_kwargs = CN()
        _C.model.G.synthesis_kwargs.encoder_kwargs.mapping_kwargs.c_dim = 0
        _C.model.G.synthesis_kwargs.encoder_kwargs.mapping_kwargs.index_dim = 0
        _C.model.G.synthesis_kwargs.encoder_kwargs.mapping_kwargs.w_dim = 512
        _C.model.G.synthesis_kwargs.encoder_kwargs.mapping_kwargs.embed_features = 512
        _C.model.G.synthesis_kwargs.encoder_kwargs.mapping_kwargs.num_layers = 2
        _C.model.G.synthesis_kwargs.encoder_kwargs.mapping_kwargs.posi_emb = 'linear'

        _C.model.G.mapping_kwargs = CN()
        _C.model.G.mapping_kwargs.num_layers = 8
        _C.model.G.mapping_kwargs.posi_emb = 'linear'

        _C.model.D.channel_base = int(0.5 * 32768)
        _C.model.D.num_fp16_res = 0
        _C.model.D.conv_clamp = None
        _C.model.D.channel_max = 512
        _C.model.D.c_dim = 0
        _C.model.D.index_dim = 0
        _C.model.D.img_resolution = 256
        _C.model.D.img_channels = 2

        _C.model.D.mapping_kwargs = CN()
        _C.model.D.mapping_kwargs.num_layers = 8
        _C.model.D.mapping_kwargs.posi_emb = 'linear'
        _C.model.D.mapping_kwargs.embed_features = 512
        _C.model.D.epilogue_kwargs = CN()
        _C.model.D.epilogue_kwargs.mbstd_group_size = 16

        _C.loss.blur_init_sigma = 0
        _C.loss.blur_fade_kimg = 0  # depend on bs
        _C.loss.lambda_consistency = 0.0
        _C.loss.lambda_sam = 0.0

        return _C

    def forward(self, update_emas=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.run_G(self.real_A, gen_index=self.gen_index, update_emas=update_emas)  # G(A)
        if self.sg:
            self.pred_mask = self.netSG(self.fake_B.unsqueeze(1)).squeeze(1)
        self.fake_B = torch.mean(self.fake_B, dim=1, keepdim=True)

    def forward_ema(self):
        self.fake_B = self.run_G_emas(self.real_A, gen_index=self.gen_index)
        self.fake_B = torch.mean(self.fake_B, dim=1, keepdim=True)
        if self.sg or (not self.isTrain and self.eval_seg):
            self.fake_B = self.fake_B.detach()
            with torch.no_grad():
                self.pred_mask = self.netSG(self.fake_B.unsqueeze(1))
                self.pred_mask = torch.sigmoid(self.pred_mask.squeeze(1))
    
    def backward_G(self, consistent=True):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        fake_AB = torch.cat((self.real_A, self.fake_B), 1) if self.combine_ab else self.fake_B
        gen_logits = self.run_D(fake_AB, c=self.gen_c, index=self.gen_index)
        self.loss_G_GAN = (torch.nn.functional.softplus(-gen_logits)).mean()
        # Second, G(A) = B
        blur_size = np.floor(self.blur_sigma * 3)

        # also blur the image when calculating L1 loss
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=fake_AB.device).div(self.blur_sigma).square().neg().exp2()
            fake_B = upfirdn2d.filter2d(self.fake_B, f / f.sum())
            real_B = upfirdn2d.filter2d(self.real_B, f / f.sum())
        else:
            fake_B = self.fake_B
            real_B = self.real_B

        self.loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.loss.lambda_L1
        if self.cross_section_consistency:
            self.run_G_emas(self.real_A_ema, self.gen_index_ema)
            if consistent:
                self.loss_G_L1 += self.criterionL1(fake_B, self.fake_B_ema) * self.opt.loss.lambda_consistency
        if self.opt.loss.lambda_vgg > 0:
            VGG_real = self.vgg(real_B.expand([int(real_B.size()[0]),3,int(real_B.size()[2]),int(real_B.size()[3])]))[0]
            VGG_fake = self.vgg(fake_B.expand([int(real_B.size()[0]),3,int(real_B.size()[2]),int(real_B.size()[3])]))[0]
            self.loss_VGG = self.criterionL1(VGG_fake, VGG_real) * self.opt.loss.lambda_vgg
            self.loss_G_L1 += self.loss_VGG
            
        if self.opt.loss.lambda_sam > 0:
            sam_in_real = (real_B.expand([int(real_B.size()[0]),3,int(real_B.size()[2]),int(real_B.size()[3])]) + 1) / 2 * 255
            sam_in_real = split_bs_to_hw(sam_in_real)
            sam_in_fake = (fake_B.expand([int(real_B.size()[0]),3,int(real_B.size()[2]),int(real_B.size()[3])]) + 1) / 2 * 255
            sam_in_fake = split_bs_to_hw(sam_in_fake)
            SAM_real = self.sam.image_encoder(self.sam.preprocess(sam_in_real))
            SAM_fake = self.sam.image_encoder(self.sam.preprocess(sam_in_fake))
            self.loss_SAM = self.criterionL1(SAM_fake, SAM_real) * self.opt.loss.lambda_sam
            self.loss_G_L1 += self.loss_SAM
        # combine loss and calculate gradientsr
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.sg:
            self.loss_seg = self.criterionSeg(self.pred_mask, self.label)
            self.loss_G += self.loss_seg * self.opt.loss.lambda_SG
            self.pred_mask = torch.sigmoid(self.pred_mask)
        self.loss_G.backward()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) if self.combine_ab else self.fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator

        gen_logits = self.run_D(fake_AB.detach(), c=self.gen_c, index=self.gen_index)
        self.loss_D_fake = (torch.nn.functional.softplus(gen_logits)).mean()
        self.loss_D_fake.backward()
        # Real
        '''
        if self.extra_b:
            real_AB = self.extra_B
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1) if self.combine_ab else self.real_B
        '''
        real_AB = torch.cat((self.real_A, self.real_B), 1) if self.combine_ab else self.real_B
        real_img_tmp = real_AB.detach().requires_grad_(True)
        real_logits = self.run_D(real_img_tmp, c=self.gen_c, index=self.gen_index)
        self.loss_D_real = (torch.nn.functional.softplus(-real_logits)).mean()
        self.loss_D = self.loss_D_real
        if self.opt.loss.lambda_r1 > 0:
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp],
                                           create_graph=True, only_inputs=True)[0]
            self.loss_Dr1 = (r1_grads.square().sum([1, 2, 3])).mean() * 0.5
            self.loss_D += self.loss_Dr1 * self.opt.loss.lambda_r1

        self.loss_D.backward()
    
    def set_input(self, input):
        super(UniCOALModel, self).set_input(input)
        if self.opt.model.G.c_dim > 0:
            self.gen_c = input['B_class'].to(self.device)
        if self.opt.model.G.index_dim > 0:
            self.gen_index = input['slice_idx'].to(self.device)
        
        if self.cross_section_consistency:
            section_idx = torch.randint(0,2,[self.real_A.size()[0], 1]).bool().to(self.device)
            thickness = input['thickness'].to(self.device)
            gen_index = torch.where(section_idx, 
                                         ((self.gen_index * thickness) + thickness) / (thickness * 2), 
                                         (self.gen_index * thickness) / (thickness * 2))
            gen_index_ema = torch.where(section_idx, 
                                        (self.gen_index * thickness) / (thickness * 2),
                                        ((self.gen_index * thickness) + thickness) / (thickness * 2))
            self.gen_index = gen_index
            self.gen_index_ema = gen_index_ema
            real_A = torch.where(section_idx.unsqueeze(-1).unsqueeze(-1), self.real_A[:, :-1], self.real_A[:, 1:])
            real_A_ema = torch.where(section_idx.unsqueeze(-1).unsqueeze(-1), self.real_A[:, 1:], self.real_A[:, :-1])
            self.real_A = real_A
            self.real_A_ema = real_A_ema
            
        if self.opt.loaders.subject_consistent:
            all_gen_z = []
            for seed in input['seed']:
                random_state = np.random.RandomState(seed)
                all_gen_z.append(random_state.randn(self.opt.model.G.z_dim))
            all_gen_z = np.array(all_gen_z)
            self.gen_z = torch.Tensor(all_gen_z).to(self.device).to(torch.float32)

    def set_test_input(self, input, slice_idx, indices):
        super(UniCOALModel, self).set_test_input(input, slice_idx, indices)
        onehot = torch.zeros([self.real_A.shape[0], len(self.opt.loaders.raw_internal_path_out)]).pin_memory().to(self.device)
        onehot[:, self.opt.loaders.test.target_modality] = 1
        if self.opt.model.G.c_dim > 0:
            self.gen_c = onehot
        if self.opt.model.G.index_dim > 0:
            self.gen_index = slice_idx.to(self.device)
        
        if self.cross_section_consistency:
            section_idx = (self.gen_index >= 0) # the first slice should be processes separately
            thickness = self.thickness
            self.gen_index = torch.where(section_idx, 
                                         ((self.gen_index * thickness) + thickness) / (thickness * 2), 
                                         (self.gen_index * thickness) / (thickness * 2))
            self.real_A = torch.where(section_idx.unsqueeze(-1).unsqueeze(-1), self.real_A[:, :-1], self.real_A[:, 1:])
            
        if self.opt.loaders.subject_consistent:
            all_gen_z = []
            for seed in range(self.real_A.shape[0]):
                random_state = np.random.RandomState(1)
                all_gen_z.append(random_state.randn(self.opt.model.G.z_dim))
            all_gen_z = np.array(all_gen_z)
            self.gen_z = torch.Tensor(all_gen_z).to(self.device).to(torch.float32)

    def optimize_parameters(self, cur_nimg, **kwargs):
        # update D
        self.blur_sigma = max(1 - cur_nimg / (self.opt.loss.blur_fade_kimg * 1e3),
                         0) * self.opt.loss.blur_init_sigma if self.opt.loss.blur_fade_kimg > 0 else 0
        self.optimizer_D.zero_grad(set_to_none=True)
        self.netD.requires_grad_(True)
        self.forward(update_emas=True)  # compute fake images: G(A)
        self.backward_D()                # calculate gradients for D
        self.netD.requires_grad_(False)
        for param in self.netD.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        self.optimizer_D.step()          # update D's weights
        # update G
        self.optimizer_G.zero_grad(set_to_none=True)
        self.netG.requires_grad_(True)
        self.forward(update_emas=False)  # compute fake images: G(A)
        # self.backward_G(cur_nimg > self.opt.loss.blur_fade_kimg * 1e3)                   # calculate graidents for G
        self.backward_G()
        self.netG.requires_grad_(False)
        for param in self.netG.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        self.optimizer_G.step()             # udpate G's weights


#Extracting VGG feature maps before the 2nd maxpooling layer  
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)       
        return h_relu2

def split_bs_to_hw(tensor):
    bs = tensor.size(0)
    tensor = torch.split(tensor, int(np.sqrt(bs)), dim=0) # tuple with length (np.sqrt(bs)), each element has size (np.sqrt(bs), C, H, W)
    tensor = torch.cat(tensor, dim=2)  # convert to (np.sqrt(bs), C, H * np.sqrt(bs), W)
    tensor = torch.split(tensor, 1, dim=0)
    tensor = torch.cat(tensor, dim=3)
    return tensor



