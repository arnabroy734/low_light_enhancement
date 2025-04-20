import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pickle
from pathlib import Path
import json
import numpy as np

# Models

class DCENet(nn.Module):
    def __init__(self):
        super(DCENet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.last = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def get_recons_image(self, alphas, image):
        for i in range(8):
            alpha = alphas[:, i*3:(i+1)*3, :, :]
            image = image + alpha*(image - image**2)
        return image

    def forward(self, image):
        _, recons_image = self.generate(image)
        return recons_image

    def generate(self, image):
        out1 = self.conv1(image)
        out1 = self.relu(out1)

        out2 = self.conv2(out1)
        out2 = self.relu(out2)

        out3 = self.conv3(out2)
        out3 = self.relu(out3)

        out4 = self.conv4(out3)
        out4 = self.relu(out4)
        out4 = torch.cat((out4, out3), dim=1)

        out5 = self.conv5(out4)
        out5 = self.relu(out5)
        out5 = torch.cat((out5, out2), dim=1)

        out6 = self.conv6(out5)
        out6 = self.relu(out6)
        out6 = torch.cat((out6, out1), dim=1)

        last = self.last(out6)
        last = self.tanh(last)
        recon_image = self.get_recons_image(last, image)

        return last, recon_image

class DCENet2(nn.Module):
    def __init__(self):
        super(DCENet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.last = nn.Conv2d(in_channels=64, out_channels=36, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def get_recons_image(self, alphas, image):
        for i in range(12):
            alpha = alphas[:, i*3:(i+1)*3, :, :]
            image = image + alpha*(image - image**2)
        return image

    def forward(self, image):
        _, recons_image = self.generate(image)
        return recons_image

    def generate(self, image):
        out1 = self.conv1(image)
        out1 = self.relu(out1)

        out2 = self.conv2(out1)
        out2 = self.relu(out2)

        out3 = self.conv3(out2)
        out3 = self.relu(out3)

        out4 = self.conv4(out3)
        out4 = self.relu(out4)
        out4 = torch.cat((out4, out3), dim=1)

        out5 = self.conv5(out4)
        out5 = self.relu(out5)
        out5 = torch.cat((out5, out2), dim=1)

        out6 = self.conv6(out5)
        out6 = self.relu(out6)
        out6 = torch.cat((out6, out1), dim=1)

        last = self.last(out6)
        last = self.tanh(last)
        recon_image = self.get_recons_image(last, image)

        return last, recon_image

class DCENetNoise2Noise(nn.Module):
    def __init__(self):
        super(DCENetNoise2Noise, self).__init__()
        # DCENet parameters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.last = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # Denoising parameters
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.noise_conv1 = nn.Conv2d(3,48,3,padding='same')
        self.noise_conv2 = nn.Conv2d(48, 48, 3, padding = 'same')
        self.noise_conv3 = nn.Conv2d(48, 3, 1)
    
    def get_recons_image(self, alphas, image):
        for i in range(8):
            alpha = alphas[:, i*3:(i+1)*3, :, :]
            image = image + alpha*(image - image**2)
        return image
    
    def get_clean_image(self, image):
        _, recons_image, noise = self.forward(image)
        clean_image = torch.clamp(recons_image - noise, 0, 1)
        return clean_image

    def forward(self, image):
        # Low light block
        alphas, recons_image = self.generate(image)
        # Denoising block
        noise = self.act(self.noise_conv1(recons_image))
        noise = self.act(self.noise_conv2(noise))
        noise = self.noise_conv3(noise)
        return alphas, recons_image, noise

    def generate(self, image):
        out1 = self.conv1(image)
        out1 = self.relu(out1)

        out2 = self.conv2(out1)
        out2 = self.relu(out2)

        out3 = self.conv3(out2)
        out3 = self.relu(out3)

        out4 = self.conv4(out3)
        out4 = self.relu(out4)
        out4 = torch.cat((out4, out3), dim=1)

        out5 = self.conv5(out4)
        out5 = self.relu(out5)
        out5 = torch.cat((out5, out2), dim=1)

        out6 = self.conv6(out5)
        out6 = self.relu(out6)
        out6 = torch.cat((out6, out1), dim=1)

        last = self.last(out6)
        last = self.tanh(last)
        recon_image = self.get_recons_image(last, image)

        return last, recon_image


# Loss of DCENet

class ExposureLoss(nn.Module):
    def __init__(self, E=0.6):
        super(ExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=16, stride=16)
        self.E = E
    def forward(self, image):
        rgb_image = 0.299*image[:, 0, :, :] + 0.587*image[:, 1, :, :] + 0.114*image[:, 2, :, :] 
        pool_image = self.pool(rgb_image)
        loss = torch.mean((pool_image - self.E)**2, dim=(1,2))
        loss = torch.mean(loss)
        return loss

class ColorConsistencyLoss(nn.Module):
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
    def forward(self, image):
        mean_image = torch.mean(image, dim=(2, 3))
        loss_1 = (mean_image[:, 0] - mean_image[:, 1])**2
        loss_2 = (mean_image[:, 1] - mean_image[:, 2])**2
        loss_3 = (mean_image[:, 2] - mean_image[:, 0])**2
        return torch.mean(loss_1+loss_2+loss_3)

class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()
    def forward(self, alphas):
        grad_h = torch.abs(torch.diff(alphas, n=1, dim=2))
        grad_v = torch.abs(torch.diff(alphas, n=1, dim=3))
        loss = torch.mean((grad_h[:,:,:,:-1] + grad_v[:,:,:-1,:])**2, dim=(2,3))
        return torch.mean(loss)

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    def forward(self, image):
        # rgb_image = 0.299*image[:, 0, :, :] + 0.587*image[:, 1, :, :] + 0.114*image[:, 2, :, :] 
        grad_h = torch.abs(torch.diff(image, n=3, dim=2))
        grad_v = torch.abs(torch.diff(image, n=3, dim=3))
        loss = torch.mean((grad_h[:,:,:,:-3] + grad_v[:,:,:-3,:])**2, dim=(2,3))
        return torch.mean(loss)

class SpatialConsistencyLoss(nn.Module):
    def __init__(self, device):
        super(SpatialConsistencyLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.left_kernel = torch.tensor([
                [0,0,0],
                [-1,1,0],
                [0,0,0]
            ], dtype=torch.float32, device=device)
        self.right_kernel = torch.tensor([
            [0,0,0],
            [0,1,-1],
            [0,0,0]
            ], dtype=torch.float32, device=device)
        self.up_kernel = torch.tensor([
            [0,-1,0],
            [0,1,0],
            [0,0,0]
            ], dtype=torch.float32, device=device)
        self.down_kernel = torch.tensor([
            [0,0,0],
            [0,1,0],
            [0,-1,0]
            ], dtype=torch.float32, device=device)
        self.left_kernel = self.left_kernel.unsqueeze(0).unsqueeze(0)
        self.right_kernel = self.right_kernel.unsqueeze(0).unsqueeze(0)
        self.up_kernel = self.up_kernel.unsqueeze(0).unsqueeze(0)
        self.down_kernel = self.down_kernel.unsqueeze(0).unsqueeze(0)


    def forward(self, Y, I):
        Y_rgb = torch.mean(Y, dim=1, keepdim=True)
        I_rgb = torch.mean(I, dim=1, keepdim=True)
        Y = self.pool(Y_rgb)
        I = self.pool(I_rgb)

        Y_left = torch.abs(F.conv2d(Y, self.left_kernel, padding='same'))
        Y_right = torch.abs(F.conv2d(Y, self.right_kernel, padding='same'))
        Y_up = torch.abs(F.conv2d(Y, self.up_kernel, padding='same'))
        Y_down = torch.abs(F.conv2d(Y, self.down_kernel, padding='same'))

        I_left = torch.abs(F.conv2d(I, self.left_kernel, padding='same'))
        I_right = torch.abs(F.conv2d(I, self.right_kernel, padding='same'))
        I_up = torch.abs(F.conv2d(I, self.up_kernel, padding='same'))
        I_down = torch.abs(F.conv2d(I, self.down_kernel, padding='same'))

        left_diff = (Y_left - I_left)**2
        right_diff = (Y_right - I_right)**2
        up_diff = (Y_up - I_up)**2
        down_diff = (Y_down - I_down)**2

        return torch.mean(left_diff + right_diff + up_diff + down_diff)

class SSIMContrastLoss(nn.Module):
    def __init__(self, device, kernel=11, sigma=1.5):
        super(SSIMContrastLoss, self).__init__()
        self.kernel = self.get_gaussian_kernel(kernel, sigma)
        self.kernel = self.kernel.view(1, 1, kernel, kernel) 
        self.kernel = self.kernel.to(device)
        self.transform = T.Compose([
            T.Grayscale()
        ])
        self.C2 = (0.03*1)**2

    def get_gaussian_kernel(self, k, sigma):
        ax = torch.arange(k) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel 
    
    def apply_gaussian_blur(self, img):
        k = self.kernel.shape[2]
        padding = k // 2
        blurred = torch.nn.functional.conv2d(img, self.kernel, padding=padding)
        return blurred.squeeze(0).squeeze(0)
    
    def forward(self,I, Y):
        I = self.transform(I)
        Y = self.transform(Y)
        mu_img1 = self.apply_gaussian_blur(I)
        mu_img2 = self.apply_gaussian_blur(Y)
        sigma_sq_image1 = self.apply_gaussian_blur(I**2) - mu_img1**2
        sigma_sq_image2 = self.apply_gaussian_blur(Y**2) - mu_img2**2
        sigma_sq_image1 = torch.abs(sigma_sq_image1)
        sigma_sq_image2 = torch.abs(sigma_sq_image2)
        # contrsat = (2*torch.sqrt(sigma_sq_image1)*torch.sqrt(sigma_sq_image2) + self.C2)/(sigma_sq_image1 + sigma_sq_image2 + self.C2)
        contrsat = (2*sigma_sq_image1*sigma_sq_image2 + self.C2)/((sigma_sq_image1 + sigma_sq_image2)**2 + self.C2)
        contrsat = torch.mean(contrsat, dim=(2,3))
        contrsat = torch.mean(contrsat)
        
        return 1-contrsat

class CombinedDCELoss(nn.Module):
    def __init__(self, device, param):
        super(CombinedDCELoss, self).__init__()
        self.param = param
        self.exp_loss_fn = ExposureLoss(E=0.7)
        self.cc_loss_fn = ColorConsistencyLoss()
        self.illsm_loss_fn = IlluminationSmoothnessLoss()
        self.spcon_loss_fn = SpatialConsistencyLoss(device)

    def forward(self, alphas, low_image, recons_image):
        exp_loss = self.param['exp_w']*self.exp_loss_fn(recons_image)
        cc_loss = self.param['cc_w']*self.cc_loss_fn(recons_image)
        illsm_loss = self.param['illsm_w']*self.illsm_loss_fn(alphas)
        spcon_loss = self.param['spcon_w']*self.spcon_loss_fn(low_image, recons_image)
        return exp_loss + cc_loss + illsm_loss + spcon_loss

# Loss of Noise Models
class Noise2NoiseLoss(nn.Module):
    def __init__(self):
        super(Noise2NoiseLoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(
            self,
            y_d1, # downsampled input
            f_y_d1, # noise of y_d1
            y_d2,
            f_y_d2,
            noise_d1, # downsampled noise
            noise_d2 
        ):
        sym_loss1 = self.mse(y_d1-f_y_d1, y_d2)
        sym_loss2 = self.mse(y_d2-f_y_d2, y_d1)
        res_loss = 0.5*(sym_loss1 + sym_loss2)
        cons_loss1 = self.mse(f_y_d1, noise_d1)
        cons_loss2 = self.mse(f_y_d2, noise_d2)
        cons_loss = 0.5*(cons_loss1 + cons_loss2)
        return res_loss + cons_loss



