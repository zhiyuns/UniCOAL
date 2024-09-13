import glob
import os
import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from itertools import chain
from torch.utils.data import ConcatDataset
from .cmsr_dataset import AbstractHDF5Dataset
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from .get_util import get_logger
logger = get_logger('TrainingSetup')


class StandardProvoDataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, ref_path=None, mirror_padding=(16, 32, 32),
                 raw_internal_path_in='raw', raw_internal_path_out='raw', random_modality_in=False, extra_raw_internal_path='raw_extra', label_internal_path='label',
                 thickness=-1, slice_num=3, provo_stage=1, weight_internal_path=None, global_normalization=True):
        self.prefix = raw_internal_path_in[0]
        self.provo_stage = provo_stage
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         ref_path=ref_path,
                         mirror_padding=mirror_padding,
                         raw_internal_path_in=raw_internal_path_in,
                         raw_internal_path_out=raw_internal_path_out,
                         random_modality_in=random_modality_in,
                         extra_raw_internal_path=extra_raw_internal_path,
                         label_internal_path=label_internal_path,
                         thickness=thickness,
                         slice_num=slice_num,
                         weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)
        self.file_path = file_path.replace(self.prefix, '')

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        # assert phase == 'test'
        # phase_config = dataset_config['test']
        phase_config = dataset_config['train'] if phase == 'train' else dataset_config['test']

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        ori_path = phase_config['ori_file_path']
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = cls.traverse_nii_paths(ori_path, file_paths, prefix=dataset_config.get('raw_internal_path_in')[0])

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              ref_path=phase_config.get('ref_path', None),
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path_in=dataset_config.get('raw_internal_path_in', 'raw'),
                              raw_internal_path_out=dataset_config.get('raw_internal_path_out', 'raw'),
                              random_modality_in=dataset_config.get('random_modality_in', False),
                              extra_raw_internal_path=dataset_config.get('extra_raw_internal_path', 'raw_extra'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              thickness=dataset_config.get('thickness', -1),
                              slice_num=dataset_config.get('slice_num', 3),
                              provo_stage=dataset_config.get('provo_stage', 1),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_nii_paths(ori_path, file_paths, prefix=''):
        assert isinstance(file_paths, list)
        results = []
        all_subjects = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                each_subjects = os.listdir(file_path)
                each_subjects = [x.replace('.h5', '') for x in each_subjects]
                all_subjects.extend(each_subjects)
            else:
                with open(file_path, 'r')  as f:
                    files = f.readlines()
                all_subjects.extend([x.strip().split('/')[-1].replace('.h5', '') for x in files])

        if os.path.isdir(ori_path):
            # if file path is a directory take all nii files in that directory
            iters = [glob.glob(os.path.join(ori_path, ext)) for ext in [f'*{prefix}.nii', f'*{prefix}.nii.gz']]
            for fp in chain(*iters):
                if os.path.basename(fp).replace(f'_{prefix}.nii.gz', '').replace(f'_{prefix}.nii', '') in all_subjects:
                    results.append(fp)
        return results
    
    def create_h5_file(self, file_path):
        # img_nii, aff, hdr = self.load_volume(file_path, im_only=False, dtype='float')
        # img_nii = self.resize_image_itk(img_nii)
        # img_data, _ = self.resample_volume(img_nii, aff, [1,1,1])
        # img_data = img_data.transpose(2,1,0)
        out_dict = {}
        for raw_name in self.raw_internal_path:
            img_nii = sitk.ReadImage(file_path.replace(self.prefix, raw_name))
            img_data = sitk.GetArrayFromImage(img_nii)
            if self.provo_stage == 2:
                img_data = np.transpose(img_data, (1,0,2))
            elif self.provo_stage == 3:
                img_data = np.transpose(img_data, (2,1,0))

            out_dict[raw_name] = img_data

        return out_dict
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'

        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        data_dict = {}
        # for key in self.raw.keys():
        #     raw_transform = self.transformer.raw_transform()
        #     data_dict[key] = raw_transform(self.raw[key][raw_idx])

        raw_transform = self.transformer.raw_transform()
        modality_B = self.raw_internal_path_out[-1]
        idx_B = idx
        raw_idx_B = self.raw_slices[idx_B]
        data_dict['B'] = raw_transform(self.raw[modality_B][raw_idx_B])

        raw_transform = self.transformer.raw_transform()
        if self.phase == 'test':
            data_A = []
            for key in self.raw_internal_path_in:
                data_A.append(raw_transform(self.raw[key][raw_idx]))
            return torch.cat(data_A), torch.Tensor(np.concatenate([raw_transform(self.raw[self.raw_internal_path_in[0]][raw_idx])])), raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx_B]
            for key in self.label.keys():
                label_transform = self.transformer.label_transform()
                data_dict[key] = label_transform(self.label[key][label_idx])
            if self.weight_map is not None:
                weight_idx = self.weight_slices[idx_B]
                for key in self.weight_map.keys():
                    weight_transform = self.transformer.weight_transform()
                    data_dict[key] = weight_transform(self.weight_map[key][weight_idx])
            # return the transformed raw and label patches
            data_A = []
            idx_A = idx
            for key in self.raw_internal_path_in:
                data_A.append(raw_transform(self.raw[key][raw_idx]))
            data_dict['A'] = np.concatenate(data_A)
            data_dict['A_fake'] = np.concatenate([raw_transform(self.raw[self.raw_internal_path_in[0]][raw_idx])])
            return data_dict


class ProvoDataset(ConcatDataset):
    def __init__(self, opt, phase='test'):
        train_datasets = StandardProvoDataset.create_datasets(opt, phase=phase)
        super(ProvoDataset, self).__init__(train_datasets)
