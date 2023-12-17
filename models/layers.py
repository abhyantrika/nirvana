import math
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple, _ntuple

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union


def _calculate_fan_in_and_fan_out(tensor_size):
    dimensions = len(tensor_size)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor_size[1]
    num_output_fmaps = tensor_size[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor_size[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor_size, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor_size)
    return fan_in if mode == 'fan_in' else fan_out

class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_dim: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 block_size: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 name:str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dim = conv_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.name = name
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        assert ( in_channels * out_channels * math.prod(kernel_size)) % (groups*math.prod(block_size)) == 0
        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels* out_channels * math.prod(kernel_size)) // groups //math.prod(block_size), conv_dim, **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels* in_channels * math.prod(kernel_size)) // groups //math.prod(block_size), conv_dim, **factory_kwargs))

        if bias is not None:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'



class Conv2d(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        weight_decoder,
        bias_decoder,
        use_straight_through:bool = False,
        compress_bias:bool = False,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        block_size: _size_2_t = (8,8),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        name='',
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, weight_decoder.conv_dim, kernel_size_, stride_, padding_, dilation_,
            block_size, False, _pair(0), groups, bias, padding_mode, name, **factory_kwargs)
        self.weight_decoder = weight_decoder
        if isinstance(self.weight_decoder,nn.Identity):
            self.block_size = (1,1)
        else:
            self.block_size = self.weight_decoder.block_size
        self.bias_decoder = bias_decoder
        self.use_straight_through = use_straight_through
        self.compress_bias = compress_bias

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        out = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return out

    def forward(self, input: Tensor) -> Tensor:
        weight = StraightThrough.apply(self.weight) if self.use_straight_through else self.weight
        bias = StraightThrough.apply(self.bias) if self.use_straight_through and self.bias is not None \
                                                   and self.compress_bias else self.bias

        w_out = self.weight_decoder(weight)
        w_out = rearrange(w_out, '(b c) (b1 c1) -> (b b1) (c c1)', b1=self.block_size[0], 
        c1=self.block_size[1], b=self.out_channels//self.block_size[0])
        if self.transposed:
            w_out = w_out.reshape(self.in_channels, self.out_channels, *self.kernel_size)
        else:
            w_out = w_out.reshape(self.out_channels, self.in_channels, *self.kernel_size)

        out = self._conv_forward(input, w_out, \
                                  self.bias_decoder(bias) if bias is not None else None)

        return out

class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, weight_decoder, bias_decoder, compress_bias:bool = False,
                 use_straight_through: bool = False, bias: bool = True, name='', device=None, dtype=None, block_size=(1,1)) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decoder = weight_decoder
        self.bias_decoder = bias_decoder
        self.use_straight_through = use_straight_through
        self.compress_bias = compress_bias

        if isinstance(self.weight_decoder,nn.Identity):
            self.block_size = (1,1)
        else:
            self.block_size = self.weight_decoder.block_size

        assert (in_features * out_features) % (math.prod(self.block_size)) == 0
        conv_dim = math.prod(block_size)
        if not isinstance(self.weight_decoder, torch.nn.Identity):
            conv_dim = self.weight_decoder.conv_dim

        self.weight = Parameter(torch.empty(
            (out_features* in_features ) // math.prod(block_size), conv_dim, **factory_kwargs))

        # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.name = name
        if bias is not None:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        weight = StraightThrough.apply(self.weight) if self.use_straight_through else self.weight
        bias = StraightThrough.apply(self.bias) if self.use_straight_through and self.bias is not None \
                                                   and self.compress_bias else self.bias
        w_out = self.weight_decoder(weight)
        w_out = rearrange(w_out, '(b c) (b1 c1) -> (b b1) (c c1)', b1=self.block_size[0], 
        c1=self.block_size[1], b=self.out_features//self.block_size[0])

        # if self.name == 'layer5':
        #     print(w_out.max().item(),w_out.min().item(), weight.max().item(), weight.min().item(), self.weight_decoder.scale.item(), self.name)  
        return F.linear(input, w_out, self.bias_decoder(bias) if bias is not None else None)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def get_dft_matrix(conv_dim, channels):
    dft = torch.zeros(conv_dim,channels)
    for i in range(conv_dim):
        for j in range(channels):
            dft[i,j] = math.cos(torch.pi/channels*(i+0.5)*j)/math.sqrt(channels)
            dft[i,j] = dft[i,j]*(math.sqrt(2) if j>0 else 1)
    return dft

class BNGlobal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x - torch.mean(x,dim=0))/(torch.std(x,dim=0)+1e-8)

class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return 2*self.act(x)-1

class Sine(torch.nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class ConvDecoder(Module):
    def __init__(
        self,
        block_size: Tuple[int, ...],
        conv_dim: int,
        init_type: str,
        decode_norm: str,
        decode_matrix:str,
        std: float,
        no_shift: bool,
        device=None,
        dtype=None,
        num_hidden:int = 0,
        hidden_sizes:int = 9,
        use_bn:bool = False,
        nonlinearity:str = 'none',
        **kwargs
    ) -> None:
        super(ConvDecoder, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        channels = math.prod(block_size)
        conv_dim = channels if conv_dim == -1 else conv_dim
        self.block_size = block_size
        self.decode_matrix = decode_matrix
        self.channels = channels
        self.conv_dim = conv_dim
        self.norm = decode_norm
        self.div = 1 if decode_norm == 'none' else None
        self.num_hidden, self.use_bn =  num_hidden, use_bn
        if num_hidden>0:
            self.hidden_sizes = _ntuple(num_hidden)(hidden_sizes)
        self.no_shift = no_shift
        if num_hidden == 0:
            if decode_matrix == 'sq':
                self.scale = Parameter(torch.empty((conv_dim, channels), **factory_kwargs))
            elif decode_matrix == 'dft':
                self.scale = Parameter(torch.empty((1,channels), **factory_kwargs))
            elif decode_matrix == 'dft_fixed':
                self.scale = Parameter(torch.empty((1,channels), **factory_kwargs),requires_grad=True)
            self.shift = Parameter(torch.empty((1,conv_dim), **factory_kwargs)) if not no_shift else 0.0
            if decode_matrix != 'sq':
                self.dft = Parameter(get_dft_matrix(conv_dim, channels), requires_grad=False)
            self.reset_parameters(init_type, std)
        else:
            act_dict = {'none':torch.nn.Identity(), 'sigmoid':Sigmoid(), 'tanh':torch.nn.Tanh(),
                             'relu':torch.nn.ReLU(), 'sine':Sine(30.0)}
            self.act = act_dict[nonlinearity]
            layers = []
            inp_dim = self.conv_dim
            for l in range(num_hidden):
                out_dim = self.hidden_sizes[l]
                out_dim = self.channels if out_dim == -1 else out_dim
                layers.append(torch.nn.Linear(inp_dim,out_dim,bias=not self.no_shift))
                if use_bn:
                    layers.append(BNGlobal())
                layers.append(self.act)
                inp_dim = out_dim
            layers.append(torch.nn.Linear(inp_dim,self.channels,bias=not self.no_shift))
            self.layers = torch.nn.Sequential(*layers)
            self.reset_parameters('random')
                    

    def reset_parameters(self, init_type, std=1.0) -> None:
        if self.num_hidden == 0:
            if init_type == 'random':
                init.normal_(self.scale, std=std)
                # init.normal_(self.shift)
            elif init_type == 'constant':
                init.constant_(self.scale, std)
            elif init_type == 'value':
                self.scale.data = std
            else:
                raise Exception(f'unknown init_type {init_type}')
            if not self.no_shift:
                init.zeros_(self.shift)
        else:
            assert init_type == 'random'
            for i,layer in enumerate(self.layers.children()):
                if isinstance(layer,torch.nn.Linear):
                    w_std = (1/layer.in_features) if i==0 else (math.sqrt(6/layer.in_features)/30)
                    if i == len(list(self.layers.children()))-1:
                        w_std = std/layer.in_features
                    # torch.nn.init.constant_(layer.weight, w_std)
                    torch.nn.init.uniform_(layer.weight, -w_std, w_std)
                    if layer.bias is not None:
                        # torch.nn.init.constant_(layer.bias, w_std)
                        torch.nn.init.uniform_(layer.bias, -w_std, w_std)

    def forward(self, input: Tensor) -> Tensor:
        # assert input.dim() == 4 and input.size(2)*input.size(3)==self.channels
        # w_in = input.reshape(input.size(0),input.size(1)*input.size(2)*input.size(3)) #assume oixhw
        if self.num_hidden == 0:
            # print(self.div, input.max(),input.min(), )
            if self.decode_matrix == 'sq':
                w_out = torch.matmul(input/self.div+self.shift,self.scale)
            else:
                w_out = torch.matmul(input/self.div+self.shift,self.dft)*self.scale
        else:
            w_out = self.layers(input/self.div)

        return w_out


class DenseDecoder(Module):
    def __init__(
        self,
        init_type: str,
        decode_norm: str,
        std: float,
        no_shift: bool,
        device=None,
        dtype=None,
        num_hidden:int = 0,
        hidden_sizes:int = 9,
        use_bn:bool = False,
        nonlinearity:str = 'none',
        **kwargs
    ) -> None:
        super(DenseDecoder, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.channels = 1
        self.conv_dim = 1
        self.decode_norm = decode_norm
        self.div = 1
        self.num_hidden, self.use_bn =  num_hidden, use_bn
        if num_hidden>0:
            self.hidden_sizes = _ntuple(num_hidden)(hidden_sizes)
        self.no_shift = no_shift
        if num_hidden == 0:
            self.scale = Parameter(torch.empty((1), **factory_kwargs))
            self.shift = Parameter(torch.empty((1), **factory_kwargs)) if not no_shift else 0.0
        if num_hidden>0:
            act_dict = {'none':torch.nn.Identity(), 'sigmoid':Sigmoid(), 'tanh':torch.nn.Tanh(),
                             'relu':torch.nn.ReLU(), 'sine':Sine(30.0)}
            self.act = act_dict[nonlinearity]
            layers = []
            inp_dim = self.conv_dim
            for l in range(num_hidden):
                out_dim = self.hidden_sizes[l]
                out_dim = self.channels if out_dim == -1 else out_dim
                layers.append(torch.nn.Linear(inp_dim,out_dim,bias=not self.no_shift))
                if use_bn:
                    layers.append(BNGlobal())
                layers.append(self.act)
                inp_dim = out_dim
            layers.append(torch.nn.Linear(inp_dim,self.channels,bias=not self.no_shift))
            self.layers = torch.nn.Sequential(*layers)
            self.reset_parameters('random')
        else:
            self.reset_parameters(init_type, std)

    def reset_parameters(self, init_type, std=1.0) -> None:
        if self.num_hidden == 0:
            if init_type == 'random':
                init.normal_(self.scale, std=std)
            elif init_type == 'constant':
                init.constant_(self.scale, std)
            elif init_type == 'value':
                self.scale.data = std
            else:
                raise Exception(f'unknown init_type {init_type}')
            if not self.no_shift:
                init.zeros_(self.shift)
        else:
            assert init_type == 'random'
            for i,layer in enumerate(self.layers.children()):
                if isinstance(layer,torch.nn.Linear):
                    w_std = (1/layer.in_features) if i==0 else (math.sqrt(6/layer.in_features)/30)
                    torch.nn.init.uniform_(layer.weight, -w_std, w_std)
                    if layer.bias:
                        torch.nn.init.uniform_(layer.bias, -w_std, w_std)

    def forward(self, input: Tensor) -> Tensor:
        # assert input.dim() == 4 and input.size(2)*input.size(3)==self.channels
        if self.num_hidden == 0:
            w_out = self.scale*(input/self.div+self.shift)
        else:
            w_out = self.layers(input.reshape(-1,1)/self.div).reshape(input.size())
        return w_out


class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


