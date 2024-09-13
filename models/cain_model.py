from .styleconsistent_model import StyleConsistentModel


class CAINModel(StyleConsistentModel):
    def __init__(self, opt):
        super(CAINModel, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = StyleConsistentModel.modify_commandline_options(config)

        _C.model.G.synthesis_kwargs.depth = 3
        _C.model.G.synthesis_kwargs.n_resgroups = 5
        _C.model.G.synthesis_kwargs.n_resblocks = 12
        _C.model.G.synthesis_kwargs.reduction = 16

        return _C

