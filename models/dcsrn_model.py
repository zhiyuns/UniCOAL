import torch
from collections import OrderedDict
from .deepresolve_model import DeepResolveModel
from .utils import define_G, define_sg, get_sg_loss


class DCSRNModel(DeepResolveModel):
    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = config
        _C.model.G.input_nc = 1
        # of input image channels: 3 for RGB and 1 for grayscale
        _C.model.G.output_nc = 1
        _C.model.G.layers = 4
        _C.model.G.n_feats = 24
        # dropout for the generator
        _C.model.G.init_type = 'kaiming'
        # network initialization [normal | xavier | kaiming | orthogonal]

        return _C
