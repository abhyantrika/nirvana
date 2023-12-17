# Based on https://github.com/lucidrains/siren-pytorch
from builtins import breakpoint
from this import d
from matplotlib import use
import torch
from torch import nn
from math import sqrt

import math
import config
import numpy as np
import torch.nn.init as init

from torch.nn.parameter import Parameter
from models.layers import ConvDecoder, DenseDecoder, Conv2d, Linear
from bitEstimator import BitEstimator

from . import model_utils

torch.pi = math.pi
class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None, weight_decoder=nn.Identity(),
                 bias_decoder=nn.Identity(), block_size=(1,1), name=''):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = Linear(dim_in, dim_out, weight_decoder, bias_decoder,
                                bias=use_bias, name=name, block_size = block_size)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out





class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.conv = CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'], 
            conv_type=kargs['conv_type'])
        self.norm = nn.Identity()
        self.act = nn.GELU() #from nerv

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PosEncoding(nn.Module):
    def __init__(self, dim, num_frames, freq):
        super().__init__()
        assert dim>1
        inv_freq = torch.zeros(dim)
        inv_freq[0::2] = torch.pow(1/freq, torch.arange(dim-dim//2))
        inv_freq[1::2] = torch.pow(1/freq, torch.arange(dim//2))
        pos_vec = inv_freq.unsqueeze(1)*torch.arange(num_frames).unsqueeze(0)
        pos_vec[1::2,:] += torch.pi/2
        self.pos_encoding = Parameter(torch.sin(pos_vec).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),requires_grad=False)
        self.num_frames = num_frames
        assert self.pos_encoding.size() == (1,dim,num_frames,1,1)

    def forward(self, x):
        assert x.dim() == 4
        N, C, H, W = x.size()
        out = x.unsqueeze(2)+self.pos_encoding
        return out.reshape(N, C*self.num_frames, H, W)

class Mod(nn.Module):
    def __init__(self, dim, num_frames, freq):
        super().__init__()
        assert dim>1
        inv_freq = torch.zeros(dim)
        inv_freq[0::2] = torch.pow(1/freq, torch.arange(dim-dim//2))
        inv_freq[1::2] = torch.pow(1/freq, torch.arange(dim//2))
        pos_vec = inv_freq.unsqueeze(1)*torch.arange(num_frames).unsqueeze(0)
        pos_vec[1::2,:] += torch.pi/2
        self.pos_encoding = Parameter(torch.sin(pos_vec),requires_grad=False)
        self.num_frames = num_frames
        assert self.pos_encoding.size() == (1,dim,num_frames,1,1)

    def forward(self, x):
        assert x.dim() == 2
        N, C, H, W = x.size()
        out = x.unsqueeze(2)+self.pos_encoding
        return out.reshape(N, C*self.num_frames, H, W)

class siren_nerv(nn.Module):
    """SIREN model with NERv blocks.
    
    Note: input has to be patches. 

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (String): Activation function.
    """
    #def __init__(self, dim_in, dim_hidden, dim_out, num_layers,patch_out, w0=30.,
    #             w0_initial=30.,img_shape=None, use_bias=True, final_activation=None,up_sample=2):
    def __init__(self, conf, patch_out,up_sample=2,dim_in=2,dim_out=3,use_bias=True):
        super().__init__()

        layers = []
        self.patch_out = patch_out
        self.up_sample = up_sample #stride of upsampling
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_bias = use_bias

        self.conf_network = config.get_conf_network(conf)
        conf_dataset = config.get_conf_dataset(conf)
        conf_decoder = config.get_conf_decoder(conf)


        self.dim_hidden = self.conf_network['layer_size']
        self.num_layers = self.conf_network['num_layers']
        self.w0 = self.conf_network['w0']
        self.w0_initial = self.conf_network['w0_initial']
        self.batch_size = self.conf_network['batch_size']
        # self.mod = self.conf_network['mod']
        # self.mod_freq = self.conf_network['mod_freq']
        # assert self.batch_size % self.mod == 0 if self.mod>0 else True
        
        use_pos = self.conf_network['use_pos']

        self.final_activation = self.conf_network['final_act']


        # Create decoders, defaults to identity if not doing entropy based compression
        self.compress = config.get_conf_losses(conf)['use_entropy_reg']
        self.conf_decoder = conf_decoder
        weight_decoders, prob_models = {}, {}
        if conf_decoder['decode_type'] == 'global':
            groups = {f'layer{k+1}': 'layer' for k in range(1, self.num_layers-1)}
            if conf_decoder['compress_first']:
                groups[f'layer1'] = (f'layer1' if conf_decoder['unique_first'] else 'layer')
            else:
                groups['layer1'] = 'no_compress'
            groups[f'layer{self.num_layers}'] = (f'layer{self.num_layers}' 
                                                    if conf_decoder['unique_last'] else 'layer')
        else:
            start_idx = 0 if conf_decoder['compress_first'] else 1
            groups = {f'layer{k+1}': f'layer{k+1}' for k in range(start_idx,self.num_layers)}
            if not conf_decoder['compress_first']:
                groups['layer1'] = 'no_compress'
        self.groups = groups
        self.unique_groups = list(set(groups.values()))
        conf_blocksize, conf_convdim = conf_decoder['block_size'], conf_decoder['conv_dim']
        conf_blocksize = {k:tuple(bs) for k,bs in conf_blocksize.items()}
        conf_convdim = {k:cd for k,cd in conf_convdim.items()}
        del conf_decoder['block_size']
        del conf_decoder['conv_dim']
        conf_decoder['std'] = 1.0
        for group in self.unique_groups:
            if group == 'no_compress':
                weight_decoders[group] = nn.Identity()
                prob_models[group] = nn.Identity()
                continue
            block_size = conf_blocksize.get(group, conf_blocksize['default'])
            conv_dim = conf_convdim.get(group,conf_convdim['default'])
            weight_decoders[group] = (ConvDecoder(block_size, conv_dim, **conf_decoder) 
                                        if self.compress else nn.Identity())
            conv_dim = math.prod(block_size) if conv_dim == -1 else conv_dim
            prob_models[group] = BitEstimator(
                                conv_dim if not self.conf_network['single_prob_model'] else 1,
                                self.conf_network['symmetric_prob_model'], 
                                num_layers=self.conf_network['prob_num_layers']
                                ) if self.compress else nn.Identity()
        if self.compress:
            print('Unique groups: ', self.unique_groups)
            print('Block sizes:',{g:weight_decoders[self.groups[g]].block_size for g in self.groups if self.groups[g]!='no_compress'})

        
        self.weight_decoders = weight_decoders
        self.bias_decoder = nn.Identity() # todo: enable bias compression
        self.prob_models = prob_models
        
        self.final_activation = model_utils.get_activation(self.final_activation)

        block_size = (1,1)
        if conf_dataset['flow']=='none':
            self.init_shift = 0.0
            # self.init_shift = Parameter(torch.empty(1,2))
            # init.zeros_(self.init_shift)
        for ind in range(self.num_layers-1):
            is_first = ind == 0
            layer_w0 = self.w0_initial if is_first else self.w0
            layer_dim_in = self.dim_in if is_first else self.dim_hidden
            group_name = f'layer{ind+1}'
            if self.compress and self.groups[group_name]!='no_compress':
                block_size = self.weight_decoders[self.groups[group_name]].block_size
            else:
                block_size = (1,1)
            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=self.dim_hidden,
                w0=layer_w0,
                use_bias=self.use_bias,
                is_first=is_first,
                name=group_name,
                weight_decoder=self.weight_decoders[self.groups[group_name]],
                bias_decoder=self.bias_decoder,
                block_size=block_size
            ))

        self.expand_w = self.patch_out[0]//2
        self.expand_h = self.patch_out[1]//2
        self.expand_ch = self.conf_network['expand_ch'] #can be large

        self.expand_dims = self.expand_w * self.expand_h * self.expand_ch

        group_name = f'layer{self.num_layers}'
        if self.compress:
            block_size = self.weight_decoders[self.groups[group_name]].block_size
        # Currently doesn't support siren_mlp due to initialization
        if self.conf_network['siren_mlp']:
            self.expand = nn.Sequential( Linear(
                self.dim_hidden, 
                self.expand_dims,
                self.weight_decoders[self.groups[group_name]],
                self.bias_decoder,
                name=group_name,
                block_size=block_size)) #can be optimized. 
        else:
            # Final layer
            layers.append(SirenLayer(
                dim_in=layer_dim_in, 
                dim_out=self.expand_dims, w0=self.w0,
                use_bias=self.use_bias, 
                activation=self.final_activation,
                name=group_name,
                weight_decoder=self.weight_decoders[self.groups[group_name]],
                bias_decoder=self.bias_decoder,
                block_size=block_size))

        self.stem = nn.Sequential(*layers)

        use_last = self.conf_network['use_last']
        nerv_in = self.expand_ch
        nerv_out = self.patch_out[1] if use_last else self.batch_size*3*(self.up_sample**2)
        if self.batch_size == 1:
            assert use_pos == 'none', 'Remove positional encoding for framewise encoding with batch size 1'
        
        if use_pos == 'none':
            last_in = nerv_out//(self.up_sample**2)
            num_groups_nerv = 1
            num_groups_last = 1
            self.pre_nerv = nn.Identity()
            self.post_nerv = nn.Identity()
        elif use_pos == 'pre_nerv':
            nerv_in = self.expand_ch*self.batch_size
            nerv_out = self.patch_out[1]*self.batch_size
            last_in = self.patch_out[1]*self.batch_size//(self.up_sample**2)
            num_groups_nerv = self.batch_size if self.conf_network['groupwise_nerv'] else 1
            num_groups_last = self.batch_size if self.conf_network['groupwise_last'] else 1
            self.pre_nerv = PosEncoding(self.expand_ch, self.batch_size, self.conf_network['freq'])
            self.post_nerv = nn.Identity()
        elif use_pos == 'post_nerv':
            assert use_last, 'Not implemented post_nerv with last layer removed'
            last_in = nerv_out*self.batch_size//(self.up_sample**2)
            num_groups_nerv = 1
            num_groups_last = self.batch_size if self.conf_network['groupwise_last'] else 1
            self.pre_nerv = nn.Identity()
            self.post_nerv = PosEncoding(self.patch_out[1]//(self.up_sample**2), self.batch_size, self.conf_network['freq'])

        nerv_out = nerv_out if use_last else self.batch_size*3*(self.up_sample**2)
        self.nerv_block = nn.Sequential(nn.Conv2d(nerv_in, nerv_out, kernel_size=(3,3),stride=(1,1),padding=(1,1), groups=num_groups_nerv),\
                                        nn.PixelShuffle(self.up_sample),\
                                        nn.GELU())

        self.last_layer = nn.Conv2d(last_in,self.batch_size*3,kernel_size=(3,3),stride=(1,1),padding=(1,1), groups=num_groups_last) if use_last else nn.Identity()
        
        self.final_activation = nn.Identity() if self.final_activation is None else self.final_activation

        if self.compress:
            boundaries = self.calc_boundaries() # calculate boundaries for each layer
            self.reset_parameters(boundaries)


    def calc_boundaries(self):
        boundaries = {}
        for m in self.modules():
            if isinstance(m,SirenLayer):
                group_name = self.groups[m.linear.name]
                if group_name == 'no_compress':
                    continue
                w0 = self.w0_initial if m.is_first else self.w0
                dim_in = m.dim_in
                w_std = (1 / dim_in) if m.is_first else (sqrt(6 / dim_in) / w0)
                if group_name not in boundaries:
                    boundaries[group_name] = w_std
                else:
                    boundaries[group_name] = min(boundaries[group_name],w_std)
        return boundaries

    def reset_parameters(self, boundaries):

        for group_name in self.weight_decoders:
            if group_name == 'no_compress':
                continue
            weight_decoder = self.weight_decoders[group_name]
            decode_matrix = self.conf_decoder['decode_matrix']
            min_std = boundaries[group_name]
            boundary = self.conf_decoder['boundary']
            if decode_matrix == 'sq':
                decoder_std = min_std/(weight_decoder.conv_dim*boundary)
            elif 'dft' in decode_matrix:
                decoder_std = min_std/(torch.mean(torch.sum(torch.abs(weight_decoder.dft),dim=0)).item()*boundary)
            decoder_std = 1/boundary
            if self.conf_decoder['decode_norm']!='none':
                # decoder_std = 1
                decoder_std = min_std
            init_type = 'constant' if self.conf_decoder['num_hidden'] == 0 else 'random'
            weight_decoder.reset_parameters(init_type,decoder_std/math.prod(weight_decoder.block_size))

            # weight_decoder.reset_parameters('random',decoder_std)
            # weight_decoder.reset_parameters('constant',0.01/4)
            # weight_decoder.reset_parameters('value',0.01)
        for m in self.modules():
            if isinstance(m,SirenLayer):
                if self.conf_decoder['num_hidden']>0:
                    continue
                group_name = self.groups[m.linear.name]
                if group_name == 'no_compress':
                    continue
                weight_decoder = self.weight_decoders[group_name]
                decode_matrix = self.conf_decoder['decode_matrix']
                w0 = self.w0_initial if m.is_first else self.w0
                dim_in = m.dim_in
                w_std = (1 / dim_in) if m.is_first else (sqrt(6 / dim_in) / w0)
                if decode_matrix == 'sq':
                    boundary = w_std/torch.mean(torch.sum(torch.abs(weight_decoder.scale),dim=0))
                else:
                    boundary = w_std/torch.mean(torch.sum(torch.abs(weight_decoder.dft),dim=0, keepdim=True)*
                                                torch.abs(weight_decoder.scale))
                boundary = torch.Tensor([1.0])

                # w_out = torch.rand(m.linear.weight.size())*2*w_std+w_std
                # w_in = torch.matmul(w_out,torch.inverse(weight_decoder.scale))
                # # w_in = torch.matmul(w_out/weight_decoder.scale,torch.inverse(weight_decoder.dft))
                # m.linear.weight.data = w_in
                # # nn.init.uniform_(m.linear.weight, -boundary.item(), boundary.item())

                mult = self.conf_decoder['boundary_first'] if m.linear.name == 'layer1' else self.conf_decoder['boundary']
                print(mult,m.linear.name)
                # nn.init.normal_(m.linear.weight,std=mult)
                nn.init.uniform_(m.linear.weight, -mult, mult)
                # nn.init.uniform_(m.linear.weight, -w_std*mult, w_std*mult)
                # print(m.linear.weight.max().item(),m.linear.weight.min().item(),group_name)

    def get_latents(self):
        weights = {}
        for m in self.modules():
            if isinstance(m, Linear):
                group_name = self.groups[m.name]
                if group_name == 'no_compress':
                    continue
                weight = m.weight
                if group_name in weights:
                    weights[group_name] = torch.cat((weights[group_name],weight))
                else:
                    weights[group_name] = weight
    
        return weights


    def forward(self, x):
        # shift = self.init_shift if self.conf_network['learnable_shift'] else 0.0
        x = x+self.init_shift
        out = self.stem(x)
        if self.conf_network['siren_mlp']:
            out = self.expand(out)

        out = out.view(out.size(0),self.expand_ch,self.expand_h,self.expand_w)
        out = self.pre_nerv(out)
        out = self.nerv_block(out)
        out = self.post_nerv(out)
        out = self.last_layer(out)
        
        out = self.final_activation(out)

        #rearrange to batch_size*3*H*W
        out = out.view(out.size(0),self.batch_size,3,self.patch_out[0],self.patch_out[1])

        return out


    def apply_straight_through(self, use_straight_through=False) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                group_name = self.groups[m.name]
                if group_name == 'no_compress':
                    continue
                m.use_straight_through = use_straight_through


    def modulated_forward(self,x,modulation):
        #out = self.stem(x)
        """
            Only doing shifts as of now. 
        """

        for module in self.stem:
            x = module.linear(x)
            x = x + modulation #shift at every siren layer. 
            out = module.activation(x)

        out = self.expand(out)
        out = out.view(out.size(0),self.expand_ch,self.expand_h,self.expand_w)
        out = self.nerv_block(out)
        out = self.last_layer(out)
        
        out = self.final_activation(out)

        return out


