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
    
class Logger():
    def __init__(self, n_epochs, batches_epoch, sv_file_name):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.save_file_name = sv_file_name
        self.loss_save = {}


    def log(self, losses=None, lr=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- lr : [%04f]' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch, lr))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data.cpu().numpy()
            else:
                self.losses[loss_name] += losses[loss_name].data.cpu().numpy()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_save:

                    self.loss_save[loss_name] = []
                    self.loss_save[loss_name].append(loss/self.batch)
                    
                else:
                    self.loss_save[loss_name].append(loss/self.batch)
                #Reset losses for next epoch
                self.losses[loss_name] = 0.0

                        
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


