import torch
import numpy as np
from .comodgan_model import CoModGANModel, CN


class StyleGANVModel(CoModGANModel):
    def __init__(self, opt):
        super(StyleGANVModel, self).__init__(opt)
        self.modality_list = opt.loaders.raw_internal_path
        self.G_motion_mapping = self.netG.module.motion_mapping

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = CoModGANModel.modify_commandline_options(config)

        _C.model.G.max_num_frames = 128
        _C.model.G.synthesis_kwargs.global_enc = True
        _C.model.G.synthesis_kwargs.relative_enc = False
        _C.model.G.synthesis_kwargs.comod_emb = False
        _C.model.G.synthesis_kwargs.coord_emb = True
        _C.model.G.motion_v_dim = 512
        _C.model.G.motion_mapping_kwargs = CN()
        _C.model.G.motion_mapping_kwargs.motion_z_dim = 512
        _C.model.G.motion_mapping_kwargs.motion_z_distance = 8
        _C.model.G.motion_mapping_kwargs.motion_kernel_size = 11
        _C.model.G.motion_mapping_kwargs.fourier = True
        _C.model.G.motion_mapping_kwargs.motion_gen_strategy = 'conv'

        _C.model.G.motion_mapping_kwargs.time_encoder_kwargs = CN()
        _C.model.G.motion_mapping_kwargs.time_encoder_kwargs.dim = 256
        _C.model.G.motion_mapping_kwargs.time_encoder_kwargs.min_period_len = 8
        _C.model.G.motion_mapping_kwargs.time_encoder_kwargs.max_period_len = 128

        _C.model.D.num_frames = 3
        _C.model.D.num_frames_div_factor = 2
        _C.model.D.max_num_frames = 128
        _C.model.D.concat_res = 16

        return _C

    def run_G(self, cond_img, noise_mode='random'):
        thickness = self.opt.loaders.thickness
        slice_num = self.opt.loaders.slice_num
        ref_img = self.extra_B if self.extra_b else self.real_B
        ws = self.G_mapping(z=self.gen_z, c=self.gen_c, img_in=ref_img)
        if self.style_mixing_prob > 0:
            pass  # not supported
        motion = self.G_motion_mapping(self.gen_t, motion_z=None)
        gen_delta_t = self.gen_delta_t*thickness+thickness if slice_num == 3 else self.gen_delta_t*thickness
        # gen_delta_t = self.gen_delta_t + 1 if slice_num == 3 else self.gen_delta_t
        img = self.G_synthesis(ws, cond_img, motion, gen_delta_t, noise_mode=noise_mode)
        return img

    def run_D(self, img, **kwargs):
        thickness = self.opt.loaders.thickness
        slice_num = self.opt.loaders.slice_num
        gen_delta_t = self.gen_delta_t * thickness + thickness if slice_num == 3 else self.gen_delta_t * thickness
        # return self.netD(img, **kwargs)
        return self.netD(img, delta_t=gen_delta_t, **kwargs)

    def forward_ema(self):
        thickness = self.opt.loaders.thickness
        slice_num = self.opt.loaders.slice_num
        ref_img = self.extra_B if self.extra_b else self.real_B
        gen_delta_t = self.gen_delta_t * thickness + thickness if slice_num == 3 else self.gen_delta_t * thickness
        # gen_delta_t = self.gen_delta_t + 1 if slice_num == 3 else self.gen_delta_t
        self.fake_B = self.netG_ema(z=self.gen_z, c=self.gen_c, t=self.gen_t, delta_t=gen_delta_t,
                                    cond_img=self.real_A, ref_img=ref_img, noise_mode='const')  # G(A)
        if self.sg or not self.isTrain:
            self.fake_B = self.fake_B.detach()
            with torch.no_grad():
                self.pred_mask = self.netSG(self.fake_B.unsqueeze(1))
                self.pred_mask = torch.sigmoid(self.pred_mask.squeeze(1))

    def set_input(self, input):
        super(StyleGANVModel, self).set_input(input)
        self.gen_c = input['B_class'].to(self.device)
        self.gen_t = input['B_idx'].to(self.device)
        self.gen_delta_t = input['slice_idx'].to(self.device)
        if self.opt.loaders.subject_consistent:
            all_gen_z = []
            for seed in input['seed']:
                random_state = np.random.RandomState(seed)
                all_gen_z.append(random_state.randn(self.opt.model.G.z_dim))
            all_gen_z = np.array(all_gen_z)
            self.gen_z = torch.Tensor(all_gen_z).to(self.device).to(torch.float32)

    def set_test_input(self, input, slice_idx, indices):
        super(StyleGANVModel, self).set_test_input(input, slice_idx, indices)
        onehot = torch.zeros([self.real_A.shape[0], len(self.opt.loaders.raw_internal_path)]).pin_memory().to(self.device)
        onehot[:, -1] = 1
        self.gen_t = torch.Tensor([x[0].start for x in indices]).to(self.device).unsqueeze(1)
        self.gen_delta_t = slice_idx.to(self.device)

        self.gen_c = onehot.to(self.device)
        if self.opt.loaders.subject_consistent:
            all_gen_z = []
            all_motion = []

            for batch_idx in range(self.real_A.shape[0]):
                random_state = np.random.RandomState(1)
                all_gen_z.append(random_state.randn(self.opt.model.G.z_dim))
                max_traj_len = self.G_motion_mapping.get_max_traj_len(self.gen_t[batch_idx]) + self.G_motion_mapping.num_additional_codes
                all_motion.append(random_state.randn(max_traj_len, self.opt.model.G.motion_v_dim))
            all_gen_z = np.array(all_gen_z)
            all_motion = np.array(all_motion)
            self.gen_z = torch.Tensor(all_gen_z).to(self.device).to(torch.float32)
            self.gen_motion_z = torch.Tensor(all_motion).to(self.device).to(torch.float32)


