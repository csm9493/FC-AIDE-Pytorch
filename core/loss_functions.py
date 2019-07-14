import torch
import torch.nn as nn

def mse_bias(output, target):
    
    X = output
    b = target
    # E[(X - b)**2]
    loss = torch.mean((target - output)**2)
    
    return loss

def estimated_bias(output, target):
    
    Z = target[:,0]
    b = output
    sigma = target[:,1]
    # E[(Z - b)**2 - sigma**2]
    loss = torch.mean((Z - b)**2 - sigma**2)
    
    return loss

def mse_linear(output, target):

    a = output[:,0]
    b = output[:,1]
    X = target[:,0]
    Z = target[:,1]
    
    # E[(X - (aZ+b))**2]
    loss = torch.mean((X - (a*Z+b))**2)
    
    return loss

def estimated_linear(output, target):
    
    a = output[:,0]
    b = output[:,1]
    X = target[:,0]
    Z = target[:,1]
    sigma = target[:,2]
    # E[(Z - (aZ+b))**2 + 2a(sigma**2) - sigma**2]
    loss = torch.mean((Z - (a*Z+b))**2 + 2*a*(sigma**2) - sigma**2)
    
    return loss

def mse_polynomial(output, target):
    
    a = output[:,0]
    b = output[:,1]
    c = output[:,2]
    X = target[:,0]
    Z = target[:,1]
    # E[(X - (a(Z**2) +bZ + c))**2]
    loss = torch.mean((X - (a*(Z**2) + b*z + c))**2)
    
    return loss

def estimated_polynomial(output, target):
    
    a = output[:,0]
    b = output[:,1]
    c = output[:,2]
    X = target[:,0]
    Z = target[:,1]
    sigma = target[:,2]
    # E[(Z - (a(Z**2) +bZ + c))**2 - 4*a*z*(sigma**2) + 2*b*(sigma**2) - sigma**2]
    loss = torch.mean((Z - (a*(Z**2) + b*z + c))**2 - 4*a*z*(sigma**2) + 2*b*(sigma**2) -sigma**2)
    
    return loss