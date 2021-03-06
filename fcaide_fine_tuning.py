import torch
import torch.nn as nn
import torch.nn.functional as F

from core.fine_tuning import Fine_tuning

_date = '190714'
_data_root_dir = './data/'
_weight_root_dir = './weights/' 
_noise_dist = 25
_mini_batch_size = 1
_learning_rate =0.0003
_model_name = 'DnCNN' #'Unet', 'DnCNN'
_patch_size = 100 #size for cropping
_epochs = 1
_data_len = '100'
_train_name = 'trdata_20500_patch_120x120.hdf5'
_test_name = 'NIPS2018_berkeley_test_images_std25.mat'
_training_type = 'specific'
_output_type = 'linear'
_case = '_test1'
_augmented = 'Full'
_weight = '190714_FCAIDE_sup_linear_specific_std25_test_ep1.w'

_save_file_name = _date  + '_FCAIDE_FT' + '_' + _output_type+ '_' + _training_type + '_std' + str(_noise_dist) + _case

_tr_data_dir = _data_root_dir + _train_name
_te_data_dir = _data_root_dir + _test_name
_weight_loc = _weight_root_dir + _weight

if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Initialize model and train
    fcaide_ft = Fine_tuning(_date, _te_data_dir=_te_data_dir, _augmented = _augmented, _output_type=_output_type, _noise_dist=_noise_dist, _epochs=_epochs, _mini_batch_size=_mini_batch_size, _learning_rate=_learning_rate, _save_file_name=_save_file_name, weight_loc = _weight_loc, _patch_size = _patch_size)