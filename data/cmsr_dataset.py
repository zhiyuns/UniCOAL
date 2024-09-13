import glob
import os
import torch
import random
from itertools import chain
import h5py
import numpy as np
from .augment import transforms
from .utils import get_slice_builder, ConfigDataset, calculate_stats
from .get_util import get_logger
from torch.utils.data import ConcatDataset
from data.augment.transforms import CropToFixed
logger = get_logger('TrainingSetup')

GLOBAL_RANDOM_STATE = np.random.RandomState(50)


def get_cls_label(shape, idx):
    onehot = np.zeros(shape, dtype=np.float32)
    onehot[idx] = 1
    label = onehot
    return label.copy()


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 ref_path=None,
                 mirror_padding=(0, 32, 32),
                 raw_internal_path_in='raw',
                 raw_internal_path_out='raw',
                 random_modality_in=False,
                 random_modality_out=False,
                 all_hr=False,
                 extra_raw_internal_path='raw_extra',
                 label_internal_path='label',
                 input_thickness_match=False,
                 thickness=(),
                 slice_num=4,
                 weight_internal_path=None,
                 global_normalization=True):
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
        assert phase in ['train', 'val', 'test']

        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)
        self.random_modality_in = random_modality_in
        self.random_modality_out = random_modality_out
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
        self.raw_internal_path = raw_internal_path = list(set(self.raw_internal_path_in + self.raw_internal_path_out))
        self.thickness = thickness
        self.slice_num = slice_num
        self.input_thickness_match = input_thickness_match

        input_file = self.create_h5_file(file_path)

        self.raw = self.fetch_and_check(input_file, raw_internal_path)

        if global_normalization:
            stats = calculate_stats(self.raw)
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(transformer_config, stats)

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label = self.fetch_and_check(input_file, label_internal_path)

            self._check_volume_sizes(self.raw, self.label)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None

        # add cropping and padding if needed
        
        crop_transform = CropToFixed(None, centered=True, mode='constant')
        for key in self.raw.keys():
            self.raw[key] = crop_transform(self.raw[key])
        if self.label is not None:
            for key in self.label.keys():
                self.label[key] = crop_transform(self.label[key])

        print(self.raw[raw_internal_path[-1]].shape)
        label_for_builder = None if self.label is None else self.label[label_internal_path[-1]]
        slice_builder = get_slice_builder(self.raw[raw_internal_path[-1]], label_for_builder,
                                          None, slice_builder_config)
        # build slice indices for raw and label data sets

        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    @staticmethod
    def fetch_and_check(input_file, internal_paths):
        ds_dict = {}
        for each_path in internal_paths:
            # assert each_path in input_file.keys(), f'Image {each_path} not found!'
            ds = input_file[each_path][:]
            if ds.ndim == 2:
                # expand dims if 2d
                ds = np.expand_dims(ds, axis=0)
            ds_dict[each_path] = ds
        return ds_dict

    def __getitem__(self, idx):
        if len(self.thickness)>0:
            if self.phase == 'train':
                thickness = random.choice(self.thickness)
                thickness_index = self.thickness.index(thickness)
            else:
                thickness = self.thickness[0]
        else:
            thickness = -1

        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'

        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        data_dict = {}

        raw_transform = self.transformer.raw_transform()
        raw_internal_path_in = self.raw_internal_path_in
        if self.random_modality_out and self.phase == 'train':
            modality_B = random.choice(self.raw_internal_path_out)
        else:
            modality_B = self.raw_internal_path_out[-1]

        if self.random_modality_in and self.phase == 'train':
            if modality_B != self.raw_internal_path_out[-1] and (not self.all_hr):
                idx_B = int((idx // thickness) * thickness) if thickness > 0 else idx
            else:
                # case for thin modality
                idx_B = idx
            raw_idx_B = self.raw_slices[idx_B]
            data_dict['B'] = raw_transform(self.raw[modality_B][raw_idx_B])

            data_dict['B_class'] = get_cls_label(len(self.raw_internal_path_out),
                                                    self.raw_internal_path_out.index(modality_B))
        else:
            raw_internal_path_in = [x for x in self.raw_internal_path_in if x != modality_B]
            idx_B = idx
            data_dict['B'] = raw_transform(self.raw[modality_B][self.raw_slices[idx_B]])
            data_dict['B_class'] = get_cls_label(len(self.raw_internal_path_out), self.raw_internal_path_out.index(modality_B))
        B_all = []
        for key in self.raw_internal_path_out:
            raw_transform = self.transformer.raw_transform()
            B_all.append(raw_transform(self.raw[key][self.raw_slices[idx_B]]))
        data_dict['B_all'] = torch.cat(B_all)

        data_dict['B_idx'] = torch.Tensor([idx_B])
        data_dict['seed'] = np.array([self.seed])
        raw_transform = self.transformer.raw_transform()
        if self.phase == 'test':
            all_input_modalities = [raw_internal_path_in[0]] if self.random_modality_in else raw_internal_path_in
            data_A = []
            for modality_A in all_input_modalities:
                if thickness > 0:
                    idx_A = int((idx_B // thickness) * thickness)

                    raw_idx_As = []
                    for posi_idx in range(self.slice_num):
                        posi = idx_A + thickness * (posi_idx + 1 - self.slice_num // 2)
                        if posi < 0 or posi > self.patch_count - 1:
                            raw_idx_As.append(None)
                        else:
                            raw_idx_As.append(self.raw_slices[posi])
                    
                    for slice_idx in raw_idx_As:
                        if slice_idx is not None:
                            raw_transform = self.transformer.raw_transform()
                            data_A.append(raw_transform(self.raw[modality_A][slice_idx]))
                        else:
                            data_A.append(raw_transform(np.zeros(self.raw[modality_A][0:1].shape)))
                else:
                    idx_A = idx
                    image_A = raw_transform(self.raw[modality_A][self.raw_slices[idx_A]])
                    data_A.append(image_A)
            relative_posi_index = np.array([idx_B - idx_A], dtype=np.float32) / thickness if thickness > 1 else np.array([0], dtype=np.float32)
            return torch.cat(data_A), torch.Tensor(relative_posi_index), raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx_B]
            for key in self.label.keys():
                label_transform = self.transformer.label_transform()
                data_dict[key] = label_transform(self.label[key][label_idx])
            # return the transformed raw and label patches
            if self.random_modality_in:
                if self.input_thickness_match:
                    modality_A = raw_internal_path_in[thickness_index]
                else:
                    modality_A = random.choice(raw_internal_path_in)
                if thickness > 0:
                    if modality_A != self.raw_internal_path_out[-1] and (not self.all_hr):
                        idx_A = int((idx_B // thickness) * thickness)
                    else:
                        idx_rand = int(idx_B - random.randint(0, thickness - 1))
                        idx_rand = max(idx_rand, 0)
                        idx_A = idx_rand
                    
                    raw_idx_As = []
                    for posi_idx in range(self.slice_num):
                        posi = idx_A + thickness * (posi_idx + 1 - self.slice_num // 2)
                        if posi < 0 or posi > self.patch_count - 1:
                            raw_idx_As.append(None)
                        else:
                            raw_idx_As.append(self.raw_slices[posi])
                    image_A = []
                    for slice_idx in raw_idx_As:
                        if slice_idx is not None:
                            raw_transform = self.transformer.raw_transform()
                            image_A.append(raw_transform(self.raw[modality_A][slice_idx]))
                        else:
                            image_A.append(raw_transform(np.zeros(self.raw[modality_A][0:1].shape)))
                    data_dict['A'] = torch.cat(image_A)
                else:
                    idx_A = idx
                    raw_idx_A = self.raw_slices[idx_A]
                    data_dict['A'] = raw_transform(self.raw[modality_A][raw_idx_A])
            else:
                image_A = []
                for modality_A in raw_internal_path_in:
                    if thickness > 0:
                        idx_A = int((idx_B // thickness) * thickness)
                        raw_idx_As = []
                        for posi_idx in range(self.slice_num):
                            posi = idx_A + thickness * (posi_idx + 1 - self.slice_num // 2)
                            if posi < 0 or posi > self.patch_count - 1:
                                raw_idx_As.append(None)
                            else:
                                raw_idx_As.append(self.raw_slices[posi])
                        
                        for slice_idx in raw_idx_As:
                            if slice_idx is not None:
                                raw_transform = self.transformer.raw_transform()
                                image_A.append(raw_transform(self.raw[modality_A][slice_idx]))
                            else:
                                image_A.append(raw_transform(np.zeros(self.raw[modality_A][0:1].shape)))
                    else:
                        idx_A = idx
                        image_A.append(raw_transform(self.raw[modality_A][raw_idx]))
                data_dict['A'] = torch.cat(image_A)

            data_dict['A_class'] = get_cls_label(len(self.raw_internal_path_in),
                                                        self.raw_internal_path_in.index(modality_A))
            data_dict['slice_idx'] = np.array([idx_B - idx_A], dtype=np.float32) / thickness if thickness > 1 else np.array([0], dtype=np.float32)
            data_dict['thickness'] = np.array([thickness], dtype=np.float32)
            return data_dict

    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_h5_file(file_path):
        raise NotImplementedError

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        for each_raw in raw.values():
            for each_label in label.values():
                assert each_raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
                assert each_label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
                assert _volume_shape(each_raw) == _volume_shape(each_label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config['train'] if phase == 'train' else dataset_config['test']

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
                              ref_path=phase_config.get('ref_path', None),
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path_in=dataset_config.get('raw_internal_path_in', 'raw'),
                              raw_internal_path_out=dataset_config.get('raw_internal_path_out', 'raw'),
                              random_modality_in=dataset_config.get('random_modality_in', False),
                              random_modality_out=dataset_config.get('random_modality_out', False),
                              all_hr=dataset_config.get('all_hr', False),
                              input_thickness_match=dataset_config.get('input_thickness_match', False),
                              extra_raw_internal_path=dataset_config.get('extra_raw_internal_path', 'raw_extra'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              thickness=dataset_config.get('thickness', ()),
                              slice_num=dataset_config.get('slice_num', 4),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            elif '.nii' in file_path:
                results.append(file_path)
            else:
                with open(file_path, 'r')  as f:
                    files = f.readlines()
                results.extend([x.strip() for x in files])
        return results


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, ref_path=None, mirror_padding=(16, 32, 32),
                 raw_internal_path_in='raw', raw_internal_path_out='raw', random_modality_in=False, random_modality_out=False,
                 all_hr=False, extra_raw_internal_path='raw_extra', label_internal_path='label',
                 thickness=(), input_thickness_match=False, slice_num=3, weight_internal_path=None, global_normalization=True):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         ref_path=ref_path,
                         mirror_padding=mirror_padding,
                         raw_internal_path_in=raw_internal_path_in,
                         raw_internal_path_out=raw_internal_path_out,
                         random_modality_in=random_modality_in,
                         random_modality_out=random_modality_out,
                         all_hr=all_hr,
                         extra_raw_internal_path=extra_raw_internal_path,
                         label_internal_path=label_internal_path,
                         thickness=thickness,
                         input_thickness_match=input_thickness_match,
                         slice_num=slice_num,
                         weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)

    @staticmethod
    def create_h5_file(file_path):
        return h5py.File(file_path, 'r')


class CrossValHDF5Dataset(StandardHDF5Dataset):
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        list_file = phase_config[phase + '_file_paths']
        wanted_all = []
        for each_file in list_file:
            with open(each_file, 'r') as file:
                wanted = file.readlines()
                wanted = [x.strip() for x in wanted]
                wanted_all.extend(wanted)

        file_paths_all = cls.traverse_h5_paths(file_paths)
        file_paths = []
        for file in file_paths_all:
            if os.path.basename(file) in wanted_all:
                file_paths.append(file)

        logger.info(f'total files for {phase} set: {len(file_paths)}...')
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
                              extra_raw_internal_path=dataset_config.get('extra_raw_internal_path', 'raw_extra'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              thickness=dataset_config.get('thickness', -1),
                              input_thickness_match=dataset_config.get('input_thickness_match', False),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets


class LazyHDF5Dataset(AbstractHDF5Dataset):
    """Implementation of the HDF5 dataset which loads the data lazily. It's slower, but has a low memory footprint."""

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, ref_path=None, mirror_padding=(16, 32, 32),
                 raw_internal_path_in='raw', raw_internal_path_out='raw', random_modality_in=False, random_modality_out=False,
                 all_hr=False, extra_raw_internal_path='raw_extra', label_internal_path='label',
                 thickness=(), input_thickness_match=False, slice_num=3, weight_internal_path=None, global_normalization=True):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         ref_path=ref_path,
                         mirror_padding=mirror_padding,
                         raw_internal_path_in=raw_internal_path_in,
                         raw_internal_path_out=raw_internal_path_out,
                         random_modality_in=random_modality_in,
                         random_modality_out=random_modality_out,
                         all_hr=all_hr,
                         extra_raw_internal_path=extra_raw_internal_path,
                         label_internal_path=label_internal_path,
                         thickness=thickness,
                         input_thickness_match=input_thickness_match,
                         slice_num=slice_num,
                         weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)

        logger.info("Using modified HDF5Dataset!")

    @staticmethod
    def create_h5_file(file_path):
        return LazyHDF5File(file_path)


class SRHDF5Dataset(AbstractHDF5Dataset):
    def __init__(self, file_path,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 raw_internal_path_in='raw',
                 raw_internal_path_out='raw',
                 extra_raw_internal_path='raw_extra',
                 label_internal_path='label',
                 weight_internal_path=None,
                 global_normalization=True):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         raw_internal_path_in=raw_internal_path_in,
                         raw_internal_path_out=raw_internal_path_out,
                         extra_raw_internal_path=extra_raw_internal_path,
                         label_internal_path=label_internal_path,
                         weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)


class LazyHDF5File:
    """Implementation of the LazyHDF5File class for the LazyHDF5Dataset."""

    def __init__(self, path, internal_path=None):
        self.path = path
        self.internal_path = internal_path
        if self.internal_path:
            with h5py.File(self.path, "r") as f:
                self.ndim = f[self.internal_path].ndim
                self.shape = f[self.internal_path].shape

    def ravel(self):
        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][:].ravel()
        return data
    
    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.internal_path:
            return LazyHDF5File(self.path, arg)

        if arg == Ellipsis:
            return LazyHDF5File(self.path, self.internal_path)

        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][arg]

        return data


class CmsrDataset(ConcatDataset):
    def __init__(self, opt, phase='train'):
        train_datasets = StandardHDF5Dataset.create_datasets(opt, phase=phase)
        super(CmsrDataset, self).__init__(train_datasets)

