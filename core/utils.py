import sys
import random
import time
import datetime
import numpy as np
import scipy.io as sio

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvF
import h5py
import random
import torch

import math
from skimage import measure
from sklearn.metrics import mean_squared_error

class TrdataLoader_supervised():

    def __init__(self,_tr_data_dir=None, _train_type = None, _std=25, _crop_size = 100):

        self.tr_data_dir = _tr_data_dir
        self.std = _std
        self.crop_size = _crop_size
        self.train_type = _train_type

        self.data = h5py.File(self.tr_data_dir, "r")
        self.clean_arr = self.data["training_images"][0]/255.
        self.num_data = self.data["training_images"].shape[1]
        
        print ('num of test patches : ', self.num_data)

    def __len__(self):
        return self.num_data
    
    def crop_img(self,img_arr, size, stride):
    
        cropped_patch_arr = []

        num_images = img_arr.shape[0]
        img_size = img_arr.shape[1]
        cropped_patch_size = size

        for idx in range(img_arr.shape[0]):
            img = img_arr[idx]

            for x_axis in range(0,img_size,stride):
                for y_axis in range(0,img_size,stride):

                    x_tmp_range = x_axis+cropped_patch_size
                    y_tmp_range = y_axis+cropped_patch_size

                    if x_tmp_range < img_size and y_tmp_range < img_size:
                        cropped_patch = img[x_axis:x_tmp_range,y_axis:y_tmp_range]
                        cropped_patch_arr.append(cropped_patch)

        print ('# of cropped patches : ', len(cropped_patch_arr), ' from ', num_images, ' images')
        returned_arr = np.asarray(cropped_patch_arr)
        return returned_arr


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        img = np.array(img)
        size = img.shape[0]

        if 'blind' in self.train_type:
            std = np.random.uniform(0,55)
        else:
            std = self.std
        noise = np.random.normal(0, std/255., (size, size))

        # Add noise and clip
        noise_img = img + noise
#         noise_img = np.clip(noise_img, 0, 1)
    
        return noise_img.astype(np.float32)

    def __getitem__(self, index):

        img = Image.fromarray((self.clean_arr[index,:,:]))
        
        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        patch = tvF.crop(img, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            patch = tvF.hflip(patch)

        # Random vertical flipping
        if random.random() > 0.5:
            patch = tvF.vflip(patch)

        # Corrupt source image
        source = self._add_noise(patch)
        target = patch
        
        source = tvF.to_tensor(source)
        target = tvF.to_tensor(target)

        target = torch.cat([target,source], dim = 0)
            
        return source, target


class TedataLoader_supervised():

    def __init__(self,_tedata_dir=None):

        self.te_data_dir = _tedata_dir

        self.data = sio.loadmat(self.te_data_dir)
        self.num_data = self.data["clean_images"].shape[0]
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        source = Image.fromarray((self.data["noisy_images"][index,:,:]/255.))
        target = Image.fromarray((self.data["clean_images"][index,:,:]/255.))

        source = tvF.to_tensor(source)
        target = tvF.to_tensor(target)
        
        target = torch.cat([target,source], dim = 0)

        return source, target
    
class TedataLoader_ft():

    def __init__(self,_img=None, _augmented_type = 'Full'):

        self.img = _img
        self.augmented_type = _augmented_type
        
        self.augmented_data()
        
    def augmented_data(self):
        if self.augmented_type == 'Full' or self.augmented_type == 'Test':
            self.tr_data = np.zeros((4, self.img.shape[0], self.img.shape[1]))
            
            self.tr_data[0] = self.img
            self.tr_data[1] = np.fliplr(self.img)
            self.tr_data[2] = np.flipud(self.img)
            self.tr_data[3] = np.fliplr(np.flipud(self.img))
        else:
            self.tr_data = np.zeros((1, self.img.shape[0], self.img.shape[1]))
            self.tr_data[0] = self.img
            
    def __len__(self):
        return self.tr_data.shape[0]

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        source = Image.fromarray((self.tr_data[index,:,:]))
        target = Image.fromarray((self.tr_data[index,:,:]))

        source = tvF.to_tensor(source)
        target = tvF.to_tensor(target)
        
        return source, target
    
class TrdataLoader_ft():

    def __init__(self, _img, _sigma, _augmented_type = 'Full', _patch_size = None):

        self.img = _img
        self.augmented_type = _augmented_type
        self.sigma = _sigma
        self.patch_size = _patch_size
       
        
        self.augmented_data()
        
    def augmented_data(self):
        if self.augmented_type == 'Full' or self.augmented_type == 'Training':
            self.tr_data = np.zeros((4, self.img.shape[0], self.img.shape[1]))
            
            self.tr_data[0] = self.img
            self.tr_data[1] = np.fliplr(self.img)
            self.tr_data[2] = np.flipud(self.img)
            self.tr_data[3] = np.fliplr(np.flipud(self.img))
        else:
            self.tr_data = np.zeros((1, self.img.shape[0], self.img.shape[1]))
            self.tr_data[0] = self.img
            
    def __len__(self):
        return self.tr_data.shape[0]

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        source = Image.fromarray((self.tr_data[index,:,:]))
        target = Image.fromarray((self.tr_data[index,:,:]))
        sigma = Image.fromarray((np.ones((self.tr_data[index,:,:].shape[0], self.tr_data[index,:,:].shape[1]))*self.sigma/255.))
        
#         i, j, h, w = transforms.RandomCrop.get_params(source, output_size=(self.crop_size, self.crop_size))
        if self.patch_size == None:
            i, j, h, w = transforms.RandomCrop.get_params(source, output_size=(self.tr_data[index,:,:].shape[0], self.tr_data[index,:,:].shape[1]))
        else:
            i, j, h, w = transforms.RandomCrop.get_params(source, output_size=(self.patch_size, self.patch_size))
        source = tvF.crop(source, i, j, h, w)
        target = tvF.crop(target, i, j, h, w)
        sigma = tvF.crop(sigma, i, j, h, w)

        source = tvF.to_tensor(source)
        target = tvF.to_tensor(target)
        sigma = tvF.to_tensor(sigma)
        
        target = torch.cat([target,sigma], dim = 0)
        
        return source, target
    
def get_PSNR(X, X_hat):

    mse = mean_squared_error(X,X_hat)
    test_PSNR = 10 * math.log10(1/mse)

    return test_PSNR

def get_SSIM(X, X_hat):

    test_SSIM = measure.compare_ssim(X, X_hat, data_range=X.max() - X.min())

    return test_SSIM


