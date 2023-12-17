import torch
import torch.nn as nn

from . import hash_grid
from .layer_utils import positional_encoders
import tinycudann as tcnn
from . import gridencoder

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.dim_in = self.cfg.data.dim_in
        self.dim_out = self.cfg.data.dim_out
        self.dim_hidden = self.cfg.network.layer_size
        self.num_layers = self.cfg.network.num_layers
        self.activation = self.cfg.network.activation

        pos_encoding_type = self.cfg.network.pos_encoding.type
        if pos_encoding_type is not None:
            pos_encoding_opt = self.cfg.network.pos_encoding
            if pos_encoding_type == 'nerf':
                print('Using NeRF pos encoding')
                sidelength = self.cfg.network.sidelength
                num_freq = pos_encoding_opt.num_frequencies
                self.positional_encoding = positional_encoders.PosEncodingNeRF(in_features=self.dim_in,
                                                                               sidelength=sidelength, \
                                                                               fn_samples=None, use_nyquist=True,
                                                                               num_freq=num_freq)
                self.in_features = self.positional_encoding.output_dim

            elif pos_encoding_type == 'hash_grid':
                print('Using hash grid encoding')
                options = self.cfg.network.pos_encoding.hash_grid_encoding

                if options.type == 'cuda' and options.binarize:
                    raise NotImplementedError('Binarization not implemented for CUDA. Use python version instead')

                if options.type == 'cuda':
                    self.positional_encoding = tcnn.Encoding(self.dim_in,
                                                             {
                                                                 'otype': 'Grid',
                                                                 'type': 'Hash',
                                                                 'n_levels': options.n_levels,
                                                                 'n_features_per_level': options.n_features_per_level,
                                                                 'log2_hashmap_size': options.log2_hashmap_size,
                                                                 'base_resolution': options.base_resolution,
                                                                 'interpolation': 'Linear',
                                                                 'per_level_scale': options.per_level_scale,
                                                             },
                                                             dtype=torch.float)
                    self.in_features = self.positional_encoding.n_output_dims

                elif options.type == 'grid_encoder':                    
                    self.positional_encoding = gridencoder.GridEncoder(input_dim=self.dim_in,\
                            num_levels=options.n_levels,level_dim=options.n_features_per_level,\
                            per_level_scale=options.per_level_scale,base_resolution=options.base_resolution,\
                            log2_hashmap_size=options.log2_hashmap_size,desired_resolution=options.finest_resolution,\
                            gridtype='hash',align_corners=False,interpolation='linear')
                    self.in_features = self.positional_encoding.output_dim

                else:
                    self.positional_encoding = hash_grid.MultiResHashGrid(dim=self.dim_in,
                                                                          binarize=options.binarize,
                                                                          n_levels=options.n_levels,
                                                                          n_features_per_level=options.n_features_per_level,
                                                                          log2_hashmap_size=options.log2_hashmap_size,
                                                                          base_resolution=options.base_resolution,
                                                                          finest_resolution=options.finest_resolution)
                    self.in_features = self.positional_encoding.output_dim

            elif pos_encoding_type == 'fourier':
                self.positional_encoding = positional_encoders.PosEncodingFourier(dim_in=self.dim_in,
                                                                                  dim_hidden=self.dim_hidden, \
                                                                                  scale=pos_encoding_opt.fourier_noise_scale,
                                                                                  mapping_size=pos_encoding_opt.fourier_mapping_size)

                self.in_features = self.positional_encoding.output_dim
            else:
                raise NotImplementedError

        else:
            self.positional_encoding = nn.Identity()
            self.in_features = self.dim_in

    def forward(self, x, **kwargs):
        raise NotImplementedError
