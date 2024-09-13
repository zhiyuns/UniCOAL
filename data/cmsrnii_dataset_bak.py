import glob
import os
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


class StandardNIIDataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, prefix, slice_builder_config, transformer_config, ref_path=None, mirror_padding=(16, 32, 32),
                 raw_internal_path_in='raw', raw_internal_path_out='raw', random_modality_in=False, extra_raw_internal_path='raw_extra', label_internal_path='label',
                 thickness=-1, slice_num=3, weight_internal_path=None, global_normalization=True):
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
        self.file_path = file_path.replace(prefix, '')

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
        prefix = phase_config.get('prefix', '_predictions0')
        file_paths = cls.traverse_nii_paths(file_paths, prefix)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              prefix=prefix,
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
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_nii_paths(file_paths, prefix=''):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in [f'*{prefix}.nii', f'*{prefix}.nii.gz']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results

    @staticmethod
    def resize_image_itk(ori_img, resamplemethod=sitk.sitkLinear):
        """
        用itk方法将原始图像resample到与目标图像一致
        :param ori_img: 原始需要对齐的itk图像
        :param target_img: 要对齐的目标itk图像
        :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻 sitkBSpline
        :return:img_res_itk: 重采样好的itk图像
        使用示范：
        import SimpleITK as sitk
        target_img = sitk.ReadImage(target_img_file)
        ori_img = sitk.ReadImage(ori_img_file)
        img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
        """
        target_Size = (256, 256, ori_img.GetSize()[2]*5)  # 目标图像大小  [x,y,z]
        target_Spacing = (1, 1, 1)  # 目标的体素块尺寸    [x,y,z]
        #target_Size = (256, 256, 116)  # 目标图像大小  [x,y,z]
        #target_Spacing = (0.75, 0.75, 1)  # 目标的体素块尺寸    [x,y,z]
        target_origin = ori_img.GetOrigin()  # 目标的起点 [x,y,z]
        target_direction = ori_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

        # itk的方法进行resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
        # 设置目标图像的信息
        resampler.SetSize(target_Size)  # 目标图像大小
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputSpacing(target_Spacing)
        # 根据需要重采样图像的情况设置不同的dype
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
        return itk_img_resampled

    def resample_volume(self, volume, aff, new_vox_size, interpolation='linear', blur=False):
        """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
        :param volume: a numpy array
        :param aff: affine matrix of the volume
        :param new_vox_size: new voxel size (3 - element numpy vector) in mm
        :param interpolation: (optional) type of interpolation. Can be 'linear' or 'nearest. Default is 'linear'.
        :return: new volume and affine matrix
        """

        pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
        new_vox_size = np.array(new_vox_size)
        factor = pixdim / new_vox_size
        sigmas = 0.25 / factor
        sigmas[factor > 1] = 0  # don't blur if upsampling
        if not blur:
            sigmas = (0,0,0)

        volume_filt = gaussian_filter(volume, sigmas)

        # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
        x = np.arange(0, volume_filt.shape[0])
        y = np.arange(0, volume_filt.shape[1])
        z = np.arange(0, volume_filt.shape[2])

        my_interpolating_function = RegularGridInterpolator((x, y, z), volume_filt, method=interpolation)

        start = - (factor - 1) / (2 * factor)
        step = 1.0 / factor
        stop = start + step * np.ceil(volume_filt.shape * factor)

        xi = np.arange(start=start[0], stop=stop[0], step=step[0])
        yi = np.arange(start=start[1], stop=stop[1], step=step[1])
        zi = np.arange(start=start[2], stop=stop[2], step=step[2])
        xi[xi < 0] = 0
        yi[yi < 0] = 0
        zi[zi < 0] = 0
        xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
        yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
        zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

        xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
        volume2 = my_interpolating_function((xig, yig, zig))

        aff2 = aff.copy()
        for c in range(3):
            aff2[:-1, c] = aff2[:-1, c] / factor[c]
        aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

        return volume2, aff2
    
    def load_volume(self, path_volume, im_only=True, squeeze=True, dtype=None):
        """
        Load volume file.
        :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
        If npz format, 1) the variable name is assumed to be 'vol_data',
        2) the volume is associated with an identity affine matrix and blank header.
        :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
        :param squeeze: (optional) whether to squeeze the volume when loading.
        :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
        :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
        The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
        :return: the volume, with corresponding affine matrix and header if im_only is False.
        """
        assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

        if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
            x = nib.load(path_volume)
            if squeeze:
                volume = np.squeeze(x.get_fdata())
            else:
                volume = x.get_fdata()
            aff = x.affine
            header = x.header
        else:  # npz
            volume = np.load(path_volume)['vol_data']
            if squeeze:
                volume = np.squeeze(volume)
            aff = np.eye(4)
            header = nib.Nifti1Header()
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)

        if im_only:
            return volume
        else:
            return volume, aff, header
    
    def create_h5_file(self, file_path):
        img_nii = sitk.ReadImage(file_path)
        # img_nii, aff, hdr = self.load_volume(file_path, im_only=False, dtype='float')
        # img_nii = self.resize_image_itk(img_nii)
        img_data = sitk.GetArrayFromImage(img_nii)
        # img_data, _ = self.resample_volume(img_nii, aff, [1,1,1])
        # img_data = img_data.transpose(2,1,0)
        out_dict = {}
        for raw_name in self.raw_internal_path:
            out_dict[raw_name] = img_data

        return out_dict


class CmsrNIIDataset(ConcatDataset):
    def __init__(self, opt, phase='test'):
        train_datasets = StandardNIIDataset.create_datasets(opt, phase=phase)
        super(CmsrNIIDataset, self).__init__(train_datasets)
