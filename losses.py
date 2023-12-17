"""
    File that defines all losses. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from utils.state_tools import StateDictOperator
from pytorch_msssim import ms_ssim, ssim
from einops import rearrange
import numpy as np

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS



class lpips_loss(nn.Module):
    def __init__(self,cfg):
        super(lpips_loss,self).__init__()
        self.cfg = cfg
        self.perceptual_criterion = LPIPS(net_type='vgg').eval().cuda()

    def forward(self, input, target):
        """
            input and target are NCHW between 0 and 1
        """
        #Need to transform to -1 to 1 from [0,1]
        input = input.clamp(0,1)
        perceptual_loss = self.perceptual_criterion(input*2-1,target*2-1)
        return perceptual_loss

class l1(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(l1, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self, input, target):
        self.loss = F.l1_loss(input, target)
        return self.loss
        
class mse(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(mse, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self, input, target):
        self.loss = F.mse_loss(input, target)
        return self.loss

def self_information(weight, prob_model, is_single_model=False, is_val=False, g=None):
    weight = (weight + torch.rand(weight.shape, generator=g).to(weight)-0.5) if not is_val else torch.round(weight)
    weight_p = weight + 0.5
    weight_n = weight - 0.5
    if not is_single_model:
        prob = prob_model(weight_p) - prob_model(weight_n)
    else:
        prob = prob_model(weight_p.reshape(-1,1))-prob_model(weight_n.reshape(-1,1))
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / np.log(2.0), 0, 50))
    return total_bits, prob
    
class entropy_reg(nn.Module):
    def __init__(self,cfg):
        super(entropy_reg, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self,latents, prob_models, single_prob_model, lambda_loss):
        bits = num_elems = 0
        for group_name in latents:
            if torch.any(torch.isnan(latents[group_name])):
                raise Exception('Weights are NaNs')
            cur_bits, prob = self_information(latents[group_name],prob_models[group_name], single_prob_model, is_val=False)
            bits += cur_bits
            num_elems += prob.size(0)
        self.loss = bits/num_elems*lambda_loss #{'ent_loss': bits/num_elems*lambda_loss}
        return self.loss, bits.float().item()/8


class weight_loss(nn.Module):
    def __init__(self,cfg):
        super(weight_loss, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self,current_model,previous_model_state_dict,w_lambda = None):

        losses = []
        mse_loss = nn.MSELoss()

        for name,param in current_model.named_parameters():
            if name in previous_model_state_dict:
                try:
                    losses.append(mse_loss(param.data,previous_model_state_dict[name]).requires_grad_(True))
                except:
                    continue

        w_loss = torch.sum(torch.stack(losses))
        if w_lambda is not None:
            w_loss = w_lambda * w_loss

        return w_loss
        


