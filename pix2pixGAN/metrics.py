from skimage.metrics import structural_similarity as ssim
import math
import numpy as np
import torch

# SSIM and PSNR calculation functions
def calculate_ssim(img1, img2):
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100  # If the images are identical
    max_pixel = 1.0  # Images are normalized between -1 and 1
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
