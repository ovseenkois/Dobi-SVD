import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class ProjectedLinear(nn.Module):
    """Exact two-factor form of an output-space projected nn.Linear."""

    def __init__(self, linear, basis, name=None):
        super().__init__()
        if name is not None:
            self.name = name
        rank = basis.shape[1]
        self.ALinear = nn.Linear(linear.in_features, rank, bias=False,
                                 device=linear.weight.device, dtype=linear.weight.dtype)
        self.BLinear = nn.Linear(rank, linear.out_features, bias=linear.bias is not None,
                                 device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            self.ALinear.weight.copy_(basis.T @ linear.weight)
            self.BLinear.weight.copy_(basis)
            if linear.bias is not None:
                self.BLinear.bias.copy_(linear.bias)

    def forward(self, x):
        return self.BLinear(self.ALinear(x))

class SVDTransformLayer(nn.Module):
    def __init__(self,gamma=None, weight = None, bias = None, name = None, device = None):
        super(SVDTransformLayer, self).__init__()
        
        if name:
            self.name = name
        
        W_T = weight.T.to(torch.float32)
        U, S, V = torch.svd_lowrank(W_T, q=int(gamma), niter = 10)
        diag_S = torch.diag(S)
        sqrt_S = torch.sqrt(diag_S)
        A_weight_T = (U @ sqrt_S).to(torch.float16)
        B_weight_T = (sqrt_S @V.T).to(torch.float16)
        if bias is None: 
            self.ALinear = nn.Linear(W_T.size(0), gamma, bias=False, device = device)
            self.BLinear = nn.Linear(gamma, W_T.size(1), bias=False, device = device)
        else:
            self.ALinear = nn.Linear(W_T.size(0), gamma, bias=False, device = device)
            self.BLinear = nn.Linear(gamma, W_T.size(1), bias=True, device = device)
            self.ALinear.bias = nn.Parameter(bias).to(device)
       
        self.ALinear.weight = nn.Parameter(A_weight_T.T.contiguous()).to(device) 
        self.BLinear.weight = nn.Parameter(B_weight_T.T.contiguous()).to(device) 
                
    def forward(self, x):
        x = self.ALinear(x)
        x = self.BLinear(x)
        return x




class SVDTransformLayer_remapping(nn.Module):
    def __init__(self, weight1 = None, weight2 = None, bias = None, name = None, device = None):
        super(SVDTransformLayer_remapping, self).__init__()
        
        if name:
            self.name = name
        
        A_weight_T = (weight1).to(torch.float16)
        B_weight_T = (weight2).to(torch.float16)
        if bias is None: 
            self.ALinear = nn.Linear(weight1.size(0), weight1.size(1), bias=False, device = device)
            self.BLinear = nn.Linear(weight2.size(0), weight2.size(1), bias=False, device = device)
        else:
            self.ALinear = nn.Linear(weight1.size(0), weight1.size(1), bias=False, device = device)
            self.BLinear = nn.Linear(weight2.size(0), weight2.size(1), bias=True, device = device)
            self.ALinear.bias = nn.Parameter(bias).to(device)
       
        self.ALinear.weight = nn.Parameter(A_weight_T.T.contiguous()).to(device) 
        self.BLinear.weight = nn.Parameter(B_weight_T.T.contiguous()).to(device) 
                
    def forward(self, x):
        x = self.ALinear(x)
        x = self.BLinear(x)
        return x
