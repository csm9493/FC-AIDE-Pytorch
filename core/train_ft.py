import torch
from torch.utils.data import DataLoader

import numpy as np
from skimage import measure
import scipy.io as sio

from .utils import TedataLoader_ft, TrdataLoader_ft, get_PSNR, get_SSIM
from .loss_functions import estimated_bias, estimated_linear, estimated_polynomial
from .logger import Logger
from .models import FC_AIDE

import time

class Train_FT(object):
    def __init__(self, img_arr, sigma, epochs, lr = 0.0003, mini_batch_size = 1, augmented_type='Full', output_type = 'linear', weight_loc = None, _lambda = 0.00003, _patch_size = None):
        
        self.img_arr = img_arr
        self.sigma = sigma
        self.epochs = epochs
        self.augmented_type = augmented_type
        self.weight_loc = weight_loc
        self.mini_batch_size = mini_batch_size
        self.learning_rate = lr
        self.output_type = output_type
        self._lambda = _lambda
        self.patch_size = _patch_size
        
#         print ('sigma : ', self.sigma )
#         print ('epoch : ', self.epochs )
#         print ('augmented_type : ', self.augmented_type )
#         print ('output_type : ', self.output_type )
#         print ('sup_name : ', self.weight_loc )
        
        self.tr_data_loader = TrdataLoader_ft(self.img_arr, self.sigma, self.augmented_type, self.patch_size)
        self.tr_data_loader = DataLoader(self.tr_data_loader, batch_size=self.mini_batch_size, shuffle=False, num_workers=0, drop_last=True)
        
        self.te_data_loader = TedataLoader_ft(self.img_arr, self.augmented_type)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=self.mini_batch_size, shuffle=False, num_workers=0, drop_last=True)

        self.logger = Logger(self.epochs, len(self.tr_data_loader))
            
        self._compile()
        
    def _compile(self):
        
        self.model = FC_AIDE(channel=1, filters = 64, num_of_layers=10, output_type = self.output_type)
        self.model.load_state_dict(torch.load(self.weight_loc))
        if self._lambda != 0.0:
            self.sup_model = FC_AIDE(channel=1, filters = 64, num_of_layers=10, output_type = self.output_type)
            self.sup_model.load_state_dict(torch.load(self.weight_loc))
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
#         print ('num of parameters : ', pytorch_total_params)
        
#         print ("load supervised model")

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.output_type == 'linear':
            self.loss = estimated_linear
        elif self.output_type == 'polynomial':
            self.loss = estimated_polynomial
        else:
            self.loss = estimated_bias
        
        self.model = self.model.cuda()
        self.sup_model = self.model.cuda()
        
    def get_X_hat(self, target, output):
        
        if self.output_type == 'linear':
            a = output[:,0]
            b = output[:,1]

            Z = target[:,0]

            X_hat = a*Z+b
            
        elif self.output_type == 'polynomial':
            a = output[:,0]
            b = output[:,1]
            c = output[:,2]

            Z = target[:,0]

            X_hat = a*(Z**2)+b*Z+c
            
        else:
            
            b = output[:,0]
            
            X_hat = b
            
        return X_hat
    
    def model_regularization(self):
        l2_reg = 0
        for ft_param, sup_param in zip(self.model.parameters(), self.sup_model.parameters()):
            l2_reg += torch.norm(ft_param - sup_param)
            
        return l2_reg
            
    
    def eval(self):
        """Evaluates denoiser on validation set."""

        with torch.no_grad():
        
            denoised_img_arr = []
        
            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                source = source.cuda()
                target = target.cuda()

                # Denoise
                output = self.model(source)

                target = target.cpu().numpy()
                output = output.cpu().numpy()
                
                X_hat = self.get_X_hat(target, output)
                
                denoised_img_arr.append(X_hat[0])
                
        if self.augmented_type == 'Full' or self.augmented_type == 'Test':
            
            for i in range(len(denoised_img_arr)):
                if i == 0:
                    augmented_img = denoised_img_arr[i]
                elif i == 1:
                    augmented_img += np.fliplr(denoised_img_arr[i])
                elif i == 2:
                    augmented_img += np.flipud(denoised_img_arr[i])
                else:
                    augmented_img += np.flipud(np.fliplr(denoised_img_arr[i]))
                    
            denoised_img = augmented_img / len(denoised_img_arr)
            
        else:
            
            denoised_img = denoised_img_arr[0]
             
        return denoised_img
   
    def train(self):
        """Trains denoiser on training set."""
        
        result_denoised_img_arr = []
        result_est_loss_arr = []
        time_arr = []

        num_batches = len(self.tr_data_loader)
        
        for epoch in range(self.epochs):
            
            if epoch == 0:
                denoised_img = self.eval()
                mean_est_loss = 0
                result_est_loss_arr.append(mean_est_loss)
                result_denoised_img_arr.append(denoised_img)

            
            est_loss_arr = []
            
            start = time.time()
            
            for batch_idx, (source, target) in enumerate(self.tr_data_loader):

                self.optim.zero_grad()

                source = source.cuda()
                target = target.cuda()

                # Denoise image
                
                
                
                source_denoised = self.model(source)
                loss = self.loss(source_denoised, target)
                
                l2_reg = self.model_regularization()
                
                loss = loss + l2_reg*self._lambda

                # Zero gradients, perform a backward pass, and update the weights

                
                loss.backward()
                self.optim.step()
                
                
                
                self.logger.log(losses = {'loss': loss}, lr = self.optim.param_groups[0]['lr'])
                
                est_loss = loss.detach().cpu().numpy()
                
                est_loss_arr.append(est_loss)
                
            denoised_img = self.eval()
            
            ft_time = time.time()-start
            time_arr.append(ft_time)
            mean_ft_time = np.mean(time_arr)
            
            mean_est_loss = np.mean(est_loss_arr)
            result_est_loss_arr.append(mean_est_loss)
            
            result_denoised_img_arr.append(denoised_img)
            
            
        return result_est_loss_arr, result_denoised_img_arr, mean_ft_time

            



