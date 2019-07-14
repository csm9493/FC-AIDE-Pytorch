import torch
import torch.nn as nn
from .layers import QED_first_layer, QED_layer, Average_layer, Residual_module


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class FC_AIDE(nn.Module):
    def __init__(self, channel = 1, filters = 64, num_of_layers=10, output_type='linear'):
        super(FC_AIDE, self).__init__()
        
        print ('FC-AIDE output type : ', output_type)
       
        self.qed_first_layer = QED_first_layer(channel, filters).cuda()
        self.avg_first_layer = Average_layer(filters)
        self.residual_module_first_layer = Residual_module(filters)
        
        self.num_layers = num_of_layers

        dilated_value = 1
        
        for layer in range (num_of_layers-1):
            self.add_module('qed_' + str(layer), QED_layer(filters, filters, dilated_value).cuda())
            self.add_module('avg_' + str(layer), Average_layer(filters))
            self.add_module('residual_module_' + str(layer), Residual_module(filters).cuda())
            dilated_value += 1
            
        self.output_avg_layer = Average_layer(filters)
        self.output_conv1 =  nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size = 1).cuda()
        self.output_prelu1 = nn.PReLU(filters,0).cuda()
        self.output_residual_module = Residual_module(filters).cuda()
        
        if output_type == 'linear':
            self.output_layer = nn.Conv2d(in_channels=filters, out_channels=2, kernel_size = 1).cuda()
        elif output_type == 'polynomial':
            self.output_layer = nn.Conv2d(in_channels=filters, out_channels=3, kernel_size = 1).cuda()
        else:
            self.output_layer = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size = 1).cuda()
        
        self.qed = AttrProxy(self, 'qed_')
        self.avg = AttrProxy(self, 'avg_')
        self.residual_module = AttrProxy(self, 'residual_module_')
        
    def forward(self, x):
        
        residual_output_arr = []
        
        qed_output = self.qed_first_layer(x)
        avg_output = self.avg_first_layer(qed_output)
        residual_output = self.residual_module_first_layer(avg_output)
        
        residual_output_arr.append(residual_output)

        for i, (qed_layer, avg_layer, residual_layer)  in enumerate(zip(self.qed, self.avg, self.residual_module)):

            qed_output = qed_layer(qed_output)
            avg_output = avg_layer(qed_output)
            residual_output = residual_layer(avg_output)
            residual_output_arr.append(residual_output)
            
            if i >= self.num_layers - 2:
                break
            
        output = self.output_avg_layer(residual_output_arr)
        output = self.output_conv1(output)
        output = self.output_prelu1(output)
        output = self.output_residual_module(output)
        output = self.output_layer(output)
        
        return output

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, output_type='linear', inplace_type = True):
        super(DnCNN, self).__init__()
        
        print ('DnCNN output_type : ', output_type)
        self.output_type = output_type
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        self.sigmoid = nn.Sigmoid() 
        self.tanh = nn.Tanh() 
        
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=inplace_type))
        
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        _input = x
        out = self.dncnn(x)
        if self.output_type == 'sigmoid':
            output = self.sigmoid(x - out)
        elif self.output_type == 'tanh':
            output = self.tanh(x - out)
        else:
            output = (x - out)
            
        return output