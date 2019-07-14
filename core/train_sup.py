import torch
from torch.utils.data import DataLoader

import math
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage import measure
import scipy.io as sio

from .utils import TrdataLoader_supervised
from .utils import TedataLoader_supervised
from .loss_functions import mse_bias, mse_linear, mse_polynomial
from .logger import Logger
from .models import FC_AIDE

class Train_sup(object):
    def __init__(self,_date, _tr_data_dir=None, _te_data_dir=None, _training_type=None,_output_type = 'linear', _noise_dist=25, _epochs=50,
                 _drop_ep = 20, _mini_batch_size=64, _learning_rate=0.001, _crop_size = 100, _save_file_name=None):
        
        self.tr_data_dir = _tr_data_dir
        self.te_data_dir = _te_data_dir
        self.training_type = _training_type
        self.std = _noise_dist
        self.mini_batch_size = _mini_batch_size
        self.learning_rate = _learning_rate
        self.epochs = _epochs
        self.crop_size = _crop_size
        self.save_file_name = _save_file_name
        self.output_type = _output_type
        self.drop_ep = _drop_ep
        
        self.channel = 1
        
        print ('tr_data_dir : ', self.tr_data_dir )
        print ('te_data_dir : ', self.te_data_dir )
        print ('training_type : ', self.training_type )
        print ('output_type : ', self.output_type )
        print ('std : ',self.std )
        print ('mini_batch_size : ', self.mini_batch_size )
        print ('learning_rate : ', self.learning_rate )
        print ('epochs : ', self.epochs )
        print ('crop_size : ', self.crop_size )
        print ('save_file_name : ', self.save_file_name )
        
              
        self.tr_data_loader = TrdataLoader_supervised(self.tr_data_dir, self.training_type,self.std,self.crop_size)
        self.tr_data_loader = DataLoader(self.tr_data_loader, batch_size=self.mini_batch_size, shuffle=True, num_workers=0, drop_last=True)

        self.te_data_loader = TedataLoader_supervised(self.te_data_dir)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_denoised_img_arr = []
        self.result_te_loss_arr = []
        self.result_tr_loss_arr = []
        self.best_psnr = 0

        self.logger = Logger(self.epochs, len(self.tr_data_loader), self.save_file_name)
            
        self._compile()
        
    def _compile(self):
        
        self.model = FC_AIDE(channel=self.channel, filters = 64, num_of_layers=10, output_type = self.output_type)

        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.drop_ep, gamma=0.5)

        if self.output_type == 'linear':
            self.loss = mse_linear
        elif self.output_type == 'polynomial':
            self.loss = mse_polynomial
        else:
            self.loss = mse_bias
        
        self.model = self.model.cuda()
        
    def get_PSNR(self, X, X_hat):
        
        mse = mean_squared_error(X,X_hat)
        test_PSNR = 10 * math.log10(1/mse)
        
        return test_PSNR
    
    def get_SSIM(self, X, X_hat):
        
        test_SSIM = measure.compare_ssim(X, X_hat, data_range=X.max() - X.min())
        
        return test_SSIM
        
    def save_model(self, epoch):

        torch.save(self.model.state_dict(), './weights/'+self.save_file_name  +'_ep'+ str(epoch) + '.w')
        return
        
    def eval(self):
        """Evaluates denoiser on validation set."""
        
        psnr_arr = []
        ssim_arr = []
        loss_arr = []
        denoised_img_arr = []

        with torch.no_grad():
        
            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                source = source.cuda()
                target = target.cuda()

                # Denoise
                output = self.model(source)

                # Update loss
                loss = self.loss(output, target)

                target = target.cpu().numpy()
                output = output.cpu().numpy()
                loss = loss.cpu().numpy()
                
                a = output[:,0]
                b = output[:,1]
                
                X = target[:,0]
                Z = target[:,1]
                
                X_hat = a*Z+b
                
                loss_arr.append(loss)
                psnr_arr.append(self.get_PSNR(X_hat[0], X[0]))
                ssim_arr.append(self.get_SSIM(X_hat[0], X[0]))
                denoised_img_arr.append(X_hat[0])
                
        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        
        if self.best_psnr <= mean_psnr:
            self.best_psnr = mean_psnr
            self.result_denoised_img_arr = denoised_img_arr.copy()
            
        return mean_loss, mean_psnr, mean_ssim
    
    def _on_epoch_end(self, epoch, mean_tr_loss):
        """Tracks and saves starts after each epoch."""
        
        mean_te_loss, mean_psnr, mean_ssim = self.eval()

        self.result_psnr_arr.append(mean_psnr)
        self.result_ssim_arr.append(mean_ssim)
        self.result_te_loss_arr.append(mean_te_loss)
        self.result_tr_loss_arr.append(mean_tr_loss)
        
        self.save_model(epoch)
        
        sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr, 'denoised_img':self.result_denoised_img_arr})

        print ('Tr loss : ', round(mean_tr_loss,4), ' Test loss : ', round(mean_te_loss,4), ' PSNR : ', round(mean_psnr,2), ' SSIM : ', round(mean_ssim,4),' Best PSNR : ', round(self.best_psnr,4)) 
            
  
    def train(self):
        """Trains denoiser on training set."""

        num_batches = len(self.tr_data_loader)
        
        for epoch in range(self.epochs):

            self.scheduler.step()
            
            tr_loss = []

            for batch_idx, (source, target) in enumerate(self.tr_data_loader):

                self.optim.zero_grad()

                source = source.cuda()
                target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)
                loss = self.loss(source_denoised, target)

                # Zero gradients, perform a backward pass, and update the weights

                loss.backward()
                self.optim.step()
                
                self.logger.log(losses = {'loss': loss}, lr = self.optim.param_groups[0]['lr'])

                tr_loss.append(loss.detach().cpu().numpy())

            mean_tr_loss = np.mean(tr_loss)
            self._on_epoch_end(epoch+1, mean_tr_loss)    

            


