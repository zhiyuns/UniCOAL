import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def ThreeD_ssim(Gimg, Limg):
    c_ssim = 0
    for i in range(Gimg.shape[0]):
        c_ssim += compare_ssim(Limg[i], Gimg[i])
    for i in range(Gimg.shape[1]):
        tempLimg = Limg[:, i, :].squeeze()
        tempGimg = Gimg[:, i, :].squeeze()
        c_ssim += compare_ssim(tempLimg, tempGimg)
    for i in range(Gimg.shape[2]):
        tempLimg = Limg[:, :, i].squeeze()
        tempGimg = Gimg[:, :, i].squeeze()
        c_ssim += compare_ssim(tempLimg, tempGimg)
    return c_ssim / sum(Gimg.shape)


def psnr_2D(Gimg, Limg):
    c_psnr = 0
    tempLimg = Limg.squeeze()
    tempGimg = Gimg.squeeze()
    # d_range = (np.max([tempLimg, tempGimg]) - np.min([tempLimg, tempGimg]))
    c_psnr += compare_psnr(tempLimg/tempLimg.max(), tempGimg/tempGimg.max())
    return c_psnr


def ThreeD_psnr(Gimg, Limg):
    c_psnr = 0
    for i in range(Gimg.shape[0]):
        tempLimg = Limg[i, :, :].squeeze()
        tempGimg = Gimg[i, :, :].squeeze()
        d_range = (np.max([tempLimg, tempGimg]) - np.min([tempLimg, tempGimg]))
        if d_range == 0:
            c_psnr += c_psnr / (i + 1)
        else:
            c_psnr += compare_psnr(tempLimg, tempGimg, data_range=d_range)
    for i in range(Gimg.shape[1]):
        tempLimg = Limg[:, i, :].squeeze()
        tempGimg = Gimg[:, i, :].squeeze()
        d_range = (np.max([tempLimg, tempGimg]) - np.min([tempLimg, tempGimg]))
        #         print(d_range)
        if d_range == 0:
            c_psnr += c_psnr / (Gimg.shape[0] + i + 1)
        else:
            c_psnr += compare_psnr(tempLimg, tempGimg, data_range=d_range)
    for i in range(Gimg.shape[2]):
        tempLimg = Limg[:, :, i].squeeze()
        tempGimg = Gimg[:, :, i].squeeze()
        d_range = (np.max([tempLimg, tempGimg]) - np.min([tempLimg, tempGimg]))
        #         print(d_range)
        if d_range == 0:
            c_psnr += c_psnr / (Gimg.shape[0] + Gimg.shape[1] + i + 1)
        else:
            c_psnr += compare_psnr(tempLimg, tempGimg, data_range=d_range)
    return c_psnr / sum(Gimg.shape)


def ThreeD_slice_psnr(Gimg, Limg, data_range=None):
    c_psnr = 0
    count = 0
    for i in range(Limg.shape[0]):
        if np.max(Limg[i]) <= 0: continue
        tempLimg = Limg[i, :, :].squeeze()
        tempGimg = Gimg[i, :, :].squeeze()
        c_psnr += compare_psnr(tempLimg, tempGimg, data_range=data_range)
        count += 1
    return c_psnr / count


def ThreeD_slice_ssim(Gimg, Limg, data_range=None):
    c_ssim = 0
    count = 0
    for i in range(Limg.shape[0]):
        if np.max(Limg[i]) <= 0: continue
        c_ssim += compare_ssim(Limg[i], Gimg[i], data_range=data_range)
        count += 1
    return c_ssim / count


def evaluate(Gimg,Limg):
    c_psnr,c_ssim,c_mse = (0,0,0)
    for i in range(Gimg.shape[0]):
        c_psnr += ThreeD_psnr(Gimg[i][0],Limg[i][0])
        c_ssim += ThreeD_ssim(Gimg[i][0],Limg[i][0])
        c_mse += skimage.measure.compare_mse(Limg[i],Gimg[i])
    return c_psnr / Gimg.shape[0], c_ssim / Gimg.shape[0], c_mse / Gimg.shape[0]


def evaluate_2D(Gimg,Limg):
    c_psnr,c_ssim,c_mse = (0,0,0)
    count = 0
    for i in range(Gimg.shape[0]):
        if np.max(Limg[i]) <= 0: continue
        c_psnr += psnr_2D(Gimg[i][0], Limg[i][0])
        c_ssim += compare_ssim(Limg[i][0].squeeze(), Gimg[i][0].squeeze())
        c_mse += np.mean(np.abs(Limg - Gimg))
        count += 1
    if count == 0:
        return None
    else:
        return c_psnr / count, c_ssim / count, c_mse / count


def evaluate_one(Gimg, Limg):
    d_range = (np.max([Gimg, Limg]) - np.min([Gimg, Limg]))
    #         c_psnr += skimage.measure.compare_psnzr(Limg[i],Gimg[i],data_range=d_range)
    c_psnr = ThreeD_psnr(Gimg, Limg)
    c_ssim = ThreeD_ssim(Gimg, Limg)
    # c_mse += compare_mse(Limg, Gimg)
    c_mae = np.mean(np.abs(Limg - Gimg))
    return c_psnr, c_ssim, c_mae

def evaluate_slice(Gimg, Limg, data_range=None):
    c_psnr = ThreeD_slice_psnr(Gimg, Limg, data_range=data_range)
    c_ssim = ThreeD_slice_ssim(Gimg, Limg, data_range=data_range)
    # c_mse += compare_mse(Limg, Gimg)
    c_mae = np.mean(np.abs(Limg - Gimg))
    return c_psnr, c_ssim, c_mae

def evaluate_3D(Gimg, Limg, data_range=None):
    c_psnr = compare_psnr(Limg, Gimg, data_range=data_range)
    c_ssim = compare_ssim(Limg, Gimg, data_range=data_range)
    c_mae = np.mean(np.abs(Limg - Gimg))
    return c_psnr, c_ssim, c_mae

def dice_one(pred, target, max_label, binary=True):
    eps = 1e-8
    if binary:
        pred = (pred > 0.5).astype('float')
        target = (target > 0.5).astype('float')
        pred = pred.astype('float')
        intersection1 = pred * target
        dice_coef = (2 * intersection1.sum() + eps) / (pred.sum() + target.sum() + eps)
        return dice_coef
    else:
        all_dice = []
        for label_id in range(1, max_label):
            pred_each = pred == label_id
            target_each = target == label_id
            pred_each = pred_each.astype('float')
            intersection1 = pred_each * target_each
            dice_coef = (2 * intersection1.sum() + eps) / (pred_each.sum() + target_each.sum() + eps)
            all_dice.append(dice_coef)
    
    return np.mean(all_dice)

def dice_brats(pred, target):
    dice1 = dice_one((pred == 3).astype('float'), (target == 3).astype('float'), 1)
    dice2 = dice_one(((pred == 1) | (pred == 3)).astype('float'), ((target == 1) | (target == 3)).astype('float'), 1)
    dice3 = dice_one((pred != 0).astype('float'), (target != 0).astype('float'), 1)

    return dice1, dice2, dice3, np.mean([dice1, dice2, dice3])

def dice(pred, target):
    N = pred.size(0)
    eps = 1e-8
    pred = (pred > 0.5).float().view(N, -1)
    target = target.view(N, -1)
    intersection1 = pred * target
    dice_coef = (2 * intersection1.sum() + eps) / (pred.sum() + target.sum() + eps)
    return dice_coef
