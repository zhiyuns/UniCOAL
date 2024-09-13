import  h5py
import os
import SimpleITK as sitk
# import nibabel as nib
from tqdm import tqdm
import numpy as np
from multiprocessing import Process

def resample_itk(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))),
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256, norm=False):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    if norm:
        volume = volume.astype(float) / (bins_num - 1)

    return volume

def convert_h5(subject, data_path, out_path):
    h5_file = os.path.join(out_path, subject+'.h5')
    h5_file = h5py.File(h5_file, 'w')
    modalities = ['T1', 'T2', 'PD']
    for modality in modalities:
        img = os.path.join(data_path, subject, subject+f'-{modality}.nii.gz')
        if os.path.exists(img):
            img_nii = sitk.ReadImage(os.path.join(data_path, subject, subject+f'-{modality}.nii.gz'))
            # reorient
            img_nii = sitk.DICOMOrient(img_nii, 'LPS')
            img_data = sitk.GetArrayFromImage(img_nii)
            img_data = rescale_intensity(img_data)
            img_data = np.around(img_data)
            img_data = np.clip(img_data, 0, 255)
            img_data = img_data.astype('uint8')

            h5_file[modality] = img_data
    h5_file.close()

data_path = r'./registrated'
out_path = r'./paired_h5'
os.makedirs(out_path, exist_ok=True)
all_subject = os.listdir(data_path)

num_processes = 5

for loop in tqdm(range(len(all_subject) // num_processes + 1)):
    processes = []
    for subject in all_subject[loop*num_processes:(loop+1)*num_processes]:
        p = Process(target=convert_h5, args=(subject,data_path, out_path))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
