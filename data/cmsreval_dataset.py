import glob
import os
import h5py
import torch
import numpy as np
import SimpleITK as sitk
from itertools import chain
from .augment import transforms
from .utils import get_slice_builder
from torch.utils.data import ConcatDataset
from .cmsr_dataset import AbstractHDF5Dataset
from .get_util import get_logger
logger = get_logger('TrainingSetup')

def get_cls_label(shape, idx):
    onehot = np.zeros(shape, dtype=np.float32)
    onehot[idx] = 1
    label = onehot
    return label.copy()

class StandardCmsrEvalDataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(0, 32, 32),
                 raw_internal_path_in='raw',
                 raw_internal_path_out='raw',
                 all_hr=False,
                 thickness=(),
                 out_thickness=1,
                 slice_num=3):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        """
        assert phase in ['test']

        self.all_hr = all_hr
        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path_in = raw_internal_path_in
        self.raw_internal_path_out = raw_internal_path_out
        raw_internal_path = self.raw_internal_path_in + self.raw_internal_path_out
        self.thickness = thickness
        self.out_thickness = out_thickness
        self.slice_num = slice_num

        input_file = self.create_h5_file(file_path)

        self.raw_in = self.fetch_and_check(input_file, raw_internal_path_in)
        example_raw = self.raw_in[self.raw_internal_path_in[0]]
        self.raw = {self.raw_internal_path_out[-1]:np.zeros([round(example_raw.shape[0]/self.out_thickness), example_raw.shape[1], example_raw.shape[2]])}

        stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(transformer_config, stats)
        # 'test' phase used only for predictions so ignore the label dataset
        self.label = None
        self.extra_raw = None
        self.weight_map = None
        self.extra_raw = None

        print(self.raw[raw_internal_path[-1]].shape)
        label_for_builder = None
        slice_builder_in = get_slice_builder(self.raw_in[raw_internal_path_in[-1]], label_for_builder,
                                          self.weight_map, slice_builder_config)
        slice_builder = get_slice_builder(self.raw[raw_internal_path_out[-1]], label_for_builder,
                                          self.weight_map, slice_builder_config)
        # build slice indices for raw and label data sets

        self.raw_slices_in = slice_builder_in.raw_slices
        
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count_in = len(self.raw_slices_in)
        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        assert phase == 'test'
        phase_config = dataset_config['test']

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = cls.traverse_h5_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path_in=dataset_config.get('raw_internal_path_in', 'raw'),
                              raw_internal_path_out=dataset_config.get('raw_internal_path_out', 'raw'),
                              thickness=dataset_config.get('thickness', -1),
                              out_thickness=dataset_config.get('out_thickness', 1),
                              slice_num=dataset_config.get('slice_num', 4))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets
    
    def __percentile_clip(self, input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=False):
        """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
        Percentiles for normalization can come from another tensor.

        Args:
            input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
                If reference_tensor is None, the percentiles from this tensor will be used.
            reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
            p_min (float, optional): Lower end percentile. Defaults to 0.5.
            p_max (float, optional): Upper end percentile. Defaults to 99.5.
            strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

        Returns:
            torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
        """
        if(reference_tensor == None):
            reference_tensor = input_tensor
        v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile

        if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
            v_min = 0
        output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
        output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]

        return output_tensor
    
    def create_h5_file(self, file_path):
        if file_path.endswith('.h5'):
            return h5py.File(file_path, 'r')
        elif file_path.endswith('.gz'):
            out_dict = {}
            for raw_name in self.raw_internal_path_in:
                # img_nii = sitk.ReadImage(glob.glob(os.path.join(file_path, file_path.split('/')[-1]+f'*{raw_name}.nii*'))[0])
                img_nii = sitk.ReadImage(file_path)
                img_data = sitk.GetArrayFromImage(img_nii)
                img_data = self.__percentile_clip(img_data)
                img_data = (img_data * 255).astype('uint8')
                # img_data = img_data[:,::2,::2]
                assert img_data.shape[1] == 256 and img_data.shape[2] == 256
                new_data = np.zeros([int(img_data.shape[0]*(self.thickness[0]/self.out_thickness)), img_data.shape[1], img_data.shape[2]])
                for idx, slice in enumerate(img_data):
                    for i in range(int(self.thickness[0]/self.out_thickness)):
                        new_data[idx*(int(self.thickness[0]/self.out_thickness))+i:idx*(int(self.thickness[0]/self.out_thickness))+i+1] = slice
                out_dict[raw_name] = new_data

            return out_dict

    
    def __getitem__(self, idx):
        if len(self.thickness)>0:
            thickness = self.thickness[0]
        else:
            thickness = -1

        out_thickness = self.out_thickness
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'

        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        data_dict = {}
        if self.slice_num == 4:
            raw_transform = self.transformer.raw_transform()

            idx_B = idx
            data_dict['B'] = raw_transform(self.raw[self.raw_internal_path_out[-1]][self.raw_slices[idx_B]])
            data_dict['B_class'] = get_cls_label(len(self.raw_internal_path_out), len(self.raw_internal_path_out) - 1)

            data_dict['B_idx'] = torch.Tensor([idx_B])
            raw_transform = self.transformer.raw_transform()
            if thickness > 0:
                idx_A = int((idx_B * out_thickness // thickness) * thickness)
                raw_idx_A_minus = self.raw_slices_in[idx_A - thickness] if idx_A - thickness >= 0 else None
                raw_idx_A = self.raw_slices_in[idx_A]
                raw_idx_A_plus = self.raw_slices_in[idx_A + thickness] if idx_A + thickness <= self.patch_count_in - 1 else None
                raw_idx_A_plus_plus = self.raw_slices_in[idx_A + thickness * 2] if idx_A + thickness * 2 <= self.patch_count_in - 1 else None
                raw_idx_As = [raw_idx_A_minus, raw_idx_A, raw_idx_A_plus, raw_idx_A_plus_plus]
                data_A = []
                for slice_idx in raw_idx_As:
                    if slice_idx is not None:
                        raw_transform = self.transformer.raw_transform()
                        data_A.append(raw_transform(self.raw_in[self.raw_internal_path_in[0]][slice_idx]))
                    else:
                        data_A.append(raw_transform(np.zeros(self.raw_in[self.raw_internal_path_in[0]][0:1].shape)))
            else:
                pass # not implemented yet
            return torch.cat(data_A), torch.Tensor(
                np.array([idx_B * out_thickness - idx_A], dtype=np.float32) / thickness), raw_idx

class CmsrEvalDataset(ConcatDataset):
    def __init__(self, opt, phase='test'):
        train_datasets = StandardCmsrEvalDataset.create_datasets(opt, phase=phase)
        super(CmsrEvalDataset, self).__init__(train_datasets)
