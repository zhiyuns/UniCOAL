import torch
import numpy as np
from .comodgan_model import CoModGANModel


class StyleConsistentModel(CoModGANModel):
    def __init__(self, opt):
        super(StyleConsistentModel, self).__init__(opt)
        self.modality_list = opt.loaders.raw_internal_path

    def set_input(self, input):
        super(StyleConsistentModel, self).set_input(input)
        if self.opt.model.G.c_dim > 0:
            self.gen_c = torch.cat((input['B_class'].to(self.device), input['slice_idx'].to(self.device)), 1)
        if self.opt.loaders.subject_consistent:
            all_gen_z = []
            for seed in input['seed']:
                random_state = np.random.RandomState(seed)
                all_gen_z.append(random_state.randn(self.opt.model.G.z_dim))
            all_gen_z = np.array(all_gen_z)
            self.gen_z = torch.Tensor(all_gen_z).to(self.device).to(torch.float32)

    def set_test_input(self, input, slice_idx, indices):
        super(StyleConsistentModel, self).set_test_input(input, slice_idx, indices)
        onehot = torch.zeros([self.real_A.shape[0], len(self.opt.loaders.raw_internal_path)]).pin_memory().to(self.device)
        onehot[:, -1] = 1
        if self.opt.model.G.c_dim > 0:
            self.gen_c = torch.cat((onehot, slice_idx.to(self.device)), 1)
        if self.opt.loaders.subject_consistent:
            all_gen_z = []
            for seed in range(self.real_A.shape[0]):
                random_state = np.random.RandomState(1)
                all_gen_z.append(random_state.randn(self.opt.model.G.z_dim))
            all_gen_z = np.array(all_gen_z)
            self.gen_z = torch.Tensor(all_gen_z).to(self.device).to(torch.float32)

