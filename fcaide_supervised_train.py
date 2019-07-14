import torch
import torch.nn as nn
import torch.nn.functional as F

from core.train_sup import Train_sup

_date = '190714'
_data_root_dir = './data/'
_weight_root_dir = './weights/' 
_noise_dist = 25
_mini_batch_size = 1
_learning_rate =0.001
_model_name = 'DnCNN' #'Unet', 'DnCNN'
_crop_size = 120 #size for cropping
_epochs = 50
_drop_ep = 10
_data_len = '100'
_train_name = 'trdata_20500_patch_120x120.hdf5'
_test_name = 'NIPS2018_berkeley_test_images_std25.mat'
_training_type = 'specific'
_output_type = 'linear'
_case = '_test'

_save_file_name = _date  + '_FCAIDE_sup' + '_' + _output_type+ '_' + _training_type + '_std' + str(_noise_dist) + _case

_tr_data_dir = _data_root_dir + _train_name
_te_data_dir = _data_root_dir + _test_name

if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Initialize model and train
    fcaide_sup = Train_sup(_date, _tr_data_dir=_tr_data_dir, _te_data_dir=_te_data_dir, _training_type = _training_type, _output_type=_output_type, _noise_dist=_noise_dist, _epochs=_epochs, _drop_ep = _drop_ep, _mini_batch_size=_mini_batch_size, _learning_rate=_learning_rate, _crop_size = _crop_size, _save_file_name=_save_file_name)
    fcaide_sup.train()