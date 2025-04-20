from PIL import Image
from torchvision import transforms as T
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

class SSIM:
    def __init__(self, kernel=7, sigma=1.5):
        self.C1 = (0.01*1)**2
        self.C2 = (0.03*1)**2
        self.C3 = self.C2/2
        self.kernel = self.get_gaussian_kernel(kernel, sigma)
        self.kernel = self.kernel.view(1, 1, kernel, kernel) 
        self.transform = T.Compose([
            T.Grayscale(),
            T.ToTensor()
        ])
    
    def get_gaussian_kernel(self, k, sigma):
        ax = torch.arange(k) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel 
    
    def apply_gaussian_blur(self, img):
        img = img.unsqueeze(0) 
        k = self.kernel.shape[2]
        padding = k // 2
        blurred = torch.nn.functional.conv2d(img, self.kernel, padding=padding)
        return blurred.squeeze(0).squeeze(0)
    
    def ssim_luminance(self, img1, img2):
        mu_img1 = self.apply_gaussian_blur(img1)
        mu_img2 = self.apply_gaussian_blur(img2)
        luminance = (2*mu_img1*mu_img2 + self.C1)/(mu_img1**2 + mu_img2**2 + self.C1)
        return luminance

    def ssim_contrast(self,img1, img2):
        mu_img1 = self.apply_gaussian_blur(img1)
        mu_img2 = self.apply_gaussian_blur(img2)
        sigma_sq_image1 = self.apply_gaussian_blur(img1**2) - mu_img1**2
        sigma_sq_image2 = self.apply_gaussian_blur(img2**2) - mu_img2**2
        sigma_sq_image1 = torch.abs(sigma_sq_image1)
        sigma_sq_image2 = torch.abs(sigma_sq_image2)
        contrsat = (2*torch.sqrt(sigma_sq_image1)*torch.sqrt(sigma_sq_image2) + self.C2)/(sigma_sq_image1 + sigma_sq_image2 + self.C2)
        return contrsat, sigma_sq_image1, sigma_sq_image2

    def ssim_structure(self, img1, img2):
        mu_img1 = self.apply_gaussian_blur(img1)
        mu_img2 = self.apply_gaussian_blur(img2)
        _, sigma_sq_image1, sigma_sq_image2 = self.ssim_contrast(img1, img2)
        structure = (self.apply_gaussian_blur(img1*img2) - mu_img1*mu_img2 + self.C3)/(torch.sqrt(sigma_sq_image1)*torch.sqrt(sigma_sq_image2) + self.C3)
        return structure
    
    def calculate_ssim(self, image1, image2):
        img1 = self.transform(image1)
        img2 = self.transform(image2)
        luminance = self.ssim_luminance(img1, img2)
        contrast, _, _ = self.ssim_contrast(img1, img2)
        structure = self.ssim_structure(img1, img2)
        return luminance, contrast, structure
