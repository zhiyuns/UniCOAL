from .base_model import BaseModel
from .pix2pix_model import Pix2PixModel
from .utils import define_sg

class DICEEVALModel(Pix2PixModel, BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.extra_b = None
        self.sg = False
        self.eval_seg = True
        self.netSG = define_sg(**opt.model.SG, gpu_ids=self.gpu_ids)
        self.netSG.eval()
        self.netSG.requires_grad_(False)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.real_A
        if not self.isTrain:
            self.fake_B = self.fake_B.detach()
        if self.sg or not self.isTrain:
            self.pred_mask = self.netSG(self.fake_B.unsqueeze(1)).squeeze(1)

