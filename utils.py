import numpy as np
import os,math,cv2
import torch
from collections import OrderedDict

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    sz = im.shape

    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def quantize(img):
    return img.clip(0, 255)
    # return img.clip(0, 255).round().astype(np.uint8)

def tensor2np(tensor, out_type=np.float32, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.float32:
        img_np = (img_np * 310.004).round(5)

    return img_np.astype(out_type)


def convert2np(tensor):
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(path):

    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit

def calculate_metrics(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # # R²
    # correlation_matrix = np.corrcoef(img1, img2)
    # r_squared = correlation_matrix[0, 1]**2

    # RMSE
    rmse = np.sqrt(((img1 - img2) ** 2).mean())

    # PSNR
    data_range = np.maximum(img1.max(), img2.max()) - np.minimum(img1.min(), img2.min())
    psnr_value = psnr(img1, img2, data_range=data_range)

    # 计算 SSIM
    ssim_value, _ = ssim(img1, img2, full=True, win_size=5,channel_axis=-1,data_range=data_range)

    return rmse, psnr_value, ssim_value
