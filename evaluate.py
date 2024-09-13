import os
import glob
import importlib
import nibabel as nib
import SimpleITK as sitk
from configs import default_argument_parser
from data.get_util import get_logger
from data.utils import get_test_loaders
from models import create_model
from util.evaluation import evaluate_3D, evaluate_slice
import numpy as np

normalize_gt = False
transpose = False
pad = 0
crop = 0
run_model = True
'''
normalize_gt = True
transpose = False
pad = 0
crop = 8
run_model = True
'''

def _get_predictor(model, output_dir, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('models.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, **predictor_config)

def __percentile_clip(input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=True):
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


def main():
    config = default_argument_parser()
    logger = get_logger('Config')
    logger.info(config)

    out_path = os.path.join(config.checkpoints_dir, config.name, 'evaluate')
    os.makedirs(out_path, exist_ok=True)
    fw = open(os.path.join(out_path, 'evaluate.txt'), 'a')
    test_loaders = get_test_loaders(config)
    if run_model:
        model = create_model(config)
        model.isTrain = False
        model.setup(config)
        predictor = _get_predictor(model, out_path, config)

    ori_path = config.loaders.test.ori_file_path
    prefix_img = '_predictions0.nii.gz'
    prefix_ori = '_ori.nii.gz'
    prefix_label = '_predictions1.nii.gz'
    prefix_input = '_predictions2.nii.gz'
    use_fets = (config.model.SG.netSG == 'FeTS')
    print(config.model.SG.netSG, use_fets, '\n\n\n\n\n')
    c_psnr = []
    c_ssim = []
    c_psnr_slice = []
    c_ssim_silce = []
    c_mae = []
    for test_loader in test_loaders:
        # run the model prediction on the test_loader and save the results in the output_dir
        if run_model:
            predictor(test_loader)
        subject = test_loader.dataset.file_path.split('/')[-1].split('.')[0]
        input = sitk.ReadImage(os.path.join(out_path, subject+prefix_input))
        img = sitk.ReadImage(os.path.join(out_path, subject+prefix_img))
        target_img = sitk.ReadImage((glob.glob(os.path.join(ori_path, subject, subject+f'*{config.loaders.raw_internal_path_out[config.loaders.test.target_modality]}.nii.gz'))+
                              glob.glob(os.path.join(ori_path, subject+f'*{config.loaders.raw_internal_path_out[config.loaders.test.target_modality]}.nii.gz')))[0])
        
        target_img = sitk.DICOMOrient(target_img, 'LPS')

        img_data = sitk.GetArrayFromImage(img)
        target_data = sitk.GetArrayFromImage(target_img)
        input_data = sitk.GetArrayFromImage(input)
        if normalize_gt:
            target_data = __percentile_clip(target_data) * 255
        if 'provo_stage' in config.loaders:
            target_data = target_data * 255.0
            if config.loaders['provo_stage'] == 2:
                img_data = img_data.transpose(1, 0, 2)
                input_data = target_data.transpose(1, 0, 2)
                label = label.transpose(1, 0, 2)
            elif config.loaders['provo_stage'] == 3:
                img_data = img_data.transpose(2, 1, 0)
                input_data = target_data.transpose(2, 1, 0)
                label = label.transpose(2, 1, 0)

        target_data = __percentile_clip(target_data)
        img_data = __percentile_clip(img_data + 1)
        
        if transpose:
            img_data = img_data.transpose(2,1,0)
            input_data = input_data.transpose(2,1,0)
        if pad:
            img_data = np.pad(img_data, ((pad,pad), (pad,pad), (0,0)), constant_values=-1)
            target_data = target_data[32:320-32,32:320-32,:]
            target_data = np.pad(target_data, ((pad,pad), (pad,pad), (0,0)), constant_values=-1)
            input_data = np.pad(input_data, ((pad,pad), (pad,pad), (0,0)), constant_values=-1)
        if crop:
            img_data = img_data[:, crop:256-crop,crop:256-crop]
            input_data = input_data[:, crop:256-crop,crop:256-crop]

        oneBEva = evaluate_3D(target_data, img_data, data_range=1)
        oneBEva_slice = evaluate_slice(target_data, img_data, data_range=1)

        c_psnr.append(oneBEva[0])
        c_ssim.append(oneBEva[1])
        c_psnr_slice.append(oneBEva_slice[0])
        c_ssim_silce.append(oneBEva_slice[1])
        c_mae.append(oneBEva[2])
        img = sitk.GetImageFromArray(input_data)
        img.CopyInformation(target_img)
        sitk.WriteImage(img, os.path.join(out_path, subject + prefix_input))
        img = sitk.GetImageFromArray(img_data)
        img.CopyInformation(target_img)
        sitk.WriteImage(img, os.path.join(out_path, subject+prefix_img))
        img = sitk.GetImageFromArray(target_data)
        img.CopyInformation(target_img)
        sitk.WriteImage(img, os.path.join(out_path, subject + prefix_ori))

        metrics = " subject:{}   psnr:{:.6}, ssim:{:.6}, psnr_slice:{:.6}, ssim_slice:{:.6}, mse:{:.6}\n".format(subject, oneBEva[0], oneBEva[1], oneBEva_slice[0], oneBEva_slice[1], oneBEva[2])
        fw.write(metrics)
        print(metrics)
        

    metrics = " ^^^VALIDATION mean psnr:{:.6}, ssim:{:.6}, psnr_slice:{:.6}, ssim_slice:{:.6}, mse:{:.6}\n".format(np.mean(c_psnr), np.mean(c_ssim), np.mean(c_psnr_slice), np.mean(c_ssim_silce), np.mean(c_mae))
    metrics += " std   psnr:{:.6}, ssim:{:.6}, psnr_slice:{:.6}, ssim_slice:{:.6}, mse:{:.6}\n".format(np.std(c_psnr), np.std(c_ssim), np.std(c_psnr_slice), np.std(c_ssim_silce), np.std(c_mae))
    fw.write(metrics)
    print(metrics)
    
if __name__ == '__main__':
    main()
