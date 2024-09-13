from torch.utils.data import ConcatDataset
from .cmsr_dataset import LazyHDF5Dataset

class CmsrLazyDataset(ConcatDataset):
    def __init__(self, opt, phase='train'):
        train_datasets = LazyHDF5Dataset.create_datasets(opt, phase=phase)
        super(CmsrLazyDataset, self).__init__(train_datasets)