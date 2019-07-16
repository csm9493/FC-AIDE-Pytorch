import torch
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio

from .utils import get_PSNR, get_SSIM
from sklearn.metrics import mean_squared_error
from .logger import Logger
from .models import FC_AIDE
from .train_ft import Train_FT

class Fine_tuning(object):
    def __init__(self,_date, _te_data_dir=None, _augmented = 'Full', _output_type = 'linear', _noise_dist=25, _epochs=30, _mini_batch_size=1, _learning_rate=0.0003, _save_file_name=None, weight_loc= None, _lambda = 0.0, _patch_size = None):
        
        self.te_data_dir = _te_data_dir
        self.std = _noise_dist
        self.mini_batch_size = _mini_batch_size
        self.learning_rate = _learning_rate
        self.epochs = _epochs
        self.save_file_name = _save_file_name
        self.output_type = _output_type
        self.sup_weight = weight_loc
        self.augmented = _augmented
        self._lambda = _lambda
        self.patch_size = _patch_size
        
        print ('te_data_dir : ', self.te_data_dir )
        print ('patch_size : ', self.patch_size )
        print ('output_type : ', self.output_type )
        print ('std : ',self.std )
        print ('mini_batch_size : ', self.mini_batch_size )
        print ('learning_rate : ', self.learning_rate )
        print ('epochs : ', self.epochs )
        print ('lambda : ', self._lambda )
        print ('save_file_name : ', self.save_file_name )
        print ('sup_weight : ', self.sup_weight )
        
        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_denoised_img_arr = []
        self.result_te_loss_arr = []
        self.result_est_loss_arr = []
        
        self.load_data()
        self.fine_tuning()
        
    def load_data(self):
        
        self.data = sio.loadmat(self.te_data_dir)
        
        self.noisy_data = self.data["noisy_images"][:,:,:]/255.
        self.clean_data = self.data["clean_images"][:,:,:]/255.
        
    def get_result_arr(self,img_idx, denoised_img_arr):
        
        psnr_arr = []
        ssim_arr = []
        mse_arr = []
        
        for i in range(len(denoised_img_arr)):
            psnr = get_PSNR(self.clean_data[img_idx], denoised_img_arr[i])
            ssim = get_SSIM(self.clean_data[img_idx], denoised_img_arr[i])
            mse = mean_squared_error(self.clean_data[img_idx],denoised_img_arr[i])
            
            psnr_arr.append(psnr)
            ssim_arr.append(ssim)
            mse_arr.append(mse)
            
        return psnr_arr, ssim_arr, mse_arr
            
    def fine_tuning(self):
        
        for img_idx in range(self.noisy_data.shape[0]):
        
            clean = self.clean_data[img_idx]
            noisy = self.noisy_data[img_idx]
        
            train_ft = Train_FT(noisy, self.std, self.epochs, augmented_type = self.augmented, output_type = self.output_type, weight_loc = self.sup_weight, _lambda = self._lambda, _patch_size = self.patch_size)
            est_loss_arr, denoised_img_arr, mean_ft_time = train_ft.train()

            psnr_arr, ssim_arr, mse_arr = self.get_result_arr(img_idx, denoised_img_arr)
            
            self.result_psnr_arr.append(psnr_arr)
            self.result_ssim_arr.append(ssim_arr)
            self.result_denoised_img_arr.append(denoised_img_arr)
            self.result_te_loss_arr.append(mse_arr)
            self.result_est_loss_arr.append(est_loss_arr)
            
            sio.savemat('./result_data/'+self.save_file_name + '_result',{'est_loss_arr':self.result_est_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr, 'denoised_img':self.result_denoised_img_arr})

            print ('IMG index : ', str(img_idx+1), ' Avg FT time per epoch : ', round(mean_ft_time,4))
            print ('Tr loss : ', round(est_loss_arr[-1],4), ' Test loss : ', round(mse_arr[-1],4), ' PSNR : ', round(psnr_arr[-1],2), ' SSIM : ', round(ssim_arr[-1],4)) 

            
  


