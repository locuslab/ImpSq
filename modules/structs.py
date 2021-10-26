import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import spectral_norm
from torch.nn.utils import weight_norm

import numpy as np

from modules.solvers import *
from modules.utils import *

class FourierFilter(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=1.):
        super().__init__()
        self.x_map = nn.Linear(in_channels, interm_channels)
        self.x_map.weight.data *= scale
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

        self.x_map_2 = nn.Linear(in_channels, interm_channels)
        self.x_map_2.weight.data *= scale
        self.x_map_2.bias.data.uniform_(-np.pi, np.pi)

        self._scale_cache = None
    
    def forward(self, x, z, use_cached=False):
        if use_cached and self._scale_cache is not None:
            scale = self._scale_cache
        else:
            scale = torch.sin(self.x_map(x))
            self._scale_cache = scale
        return scale * z

class GaborFilter(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=1., alpha=5., beta=1.):
        super().__init__()
        self.x_map = nn.Linear(in_channels, interm_channels)

        self.mu = nn.Parameter(torch.rand(interm_channels, in_channels) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((interm_channels,)))

        self.x_map.weight.data *= scale * torch.sqrt(self.gamma[:, None])
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

        self._scale_cache = None
    
    def forward(self, x, z, use_cached=False):
        if use_cached and self._scale_cache is not None:
            scale = self._scale_cache
        else:
            periodic = torch.sin(self.x_map(x))
            local = torch.exp(-0.5 * F.relu(self.gamma[None, :]) * sql2_dist(x.squeeze(0), self.mu))
            scale = periodic * local

            self._scale_cache = scale
        return scale * z

class SIRENFilter(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=1.):
        super().__init__()
        self.x_map = nn.Linear(in_channels, interm_channels)
        self.x_map.weight.data *= scale
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

        self._Wx = None
    
    def forward(self, x, z, use_cached=False):
        if use_cached and self._Wx is not None:
            Wx = self._Wx
        else:
            Wx = self.x_map(x)
            self._Wx = Wx
        return torch.sin(z + Wx)
    
class FFNFilter(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=10.):
        super().__init__()
        self.embedder = GaussianFFNEmbedder(in_channels, interm_channels // 2, scale=scale)
        self.x_map = nn.Linear(self.embedder.out_channels, interm_channels)
    
    def forward(self, x, z, use_cached=False):
        return torch.relu(z + self.x_map(self.embedder(x)))

def get_filter(filter_type, in_channels, interm_channels, scale, n_layers, init='default', **filter_options):
    n_layers = n_layers + 1
    layer_scale = np.sqrt(n_layers)

    if filter_type == 'fourier':
        return FourierFilter(in_channels, interm_channels, scale / layer_scale)
    elif filter_type == 'gabor':
        if 'alpha' in filter_options:
            alpha = filter_options['alpha']
        else:
            alpha = 6.0
        return GaborFilter(in_channels, interm_channels, scale / layer_scale, alpha / n_layers)
    elif filter_type == 'siren_like':
        return SIRENFilter(in_channels, interm_channels, scale / layer_scale)
    elif filter_type == 'relu':
        return FFNFilter(in_channels, interm_channels, 15.)
    else:
        raise ValueError("Filter {:s} not defined".format(filter_type))

class MFNLayer(nn.Module):
    def __init__(self, in_channels, interm_channels, input_scale, n_layers, filter_type='fourier', filter_options={}, bias=True, norm_type='none', init='default'):
        super().__init__()

        if norm_type == 'spectral_norm':
            self.z_map = spectral_norm(nn.Linear(interm_channels, interm_channels, bias=bias), n_power_iterations=5, target_norm=0.97)
        elif norm_type == 'weight_norm':
            self.z_map = weight_norm(nn.Linear(interm_channels, interm_channels, bias=bias), dim=0)
        elif norm_type == 'none':
            self.z_map = nn.Linear(interm_channels, interm_channels, bias=bias)
        nn.init.uniform_(self.z_map.weight, -np.sqrt(1 / interm_channels), np.sqrt(1 / interm_channels))

        self.filter_type = filter_type
        self.filter = get_filter(filter_type, in_channels, interm_channels, input_scale, n_layers, init, **filter_options)

    def forward(self, x, z, use_cached=False):
        z = self.z_map(z)
        return self.filter(x, z, use_cached)

class MFN(nn.Module):
    def __init__(self, in_channels, interm_channels, n_layers, input_scale=256., filter_type='fourier', filter_options={}, norm_type='none', init='default', weight_shared=False):
        super().__init__()
        self.weight_shared = weight_shared

        if self.weight_shared:
            self.shared_layer = MFNLayer(in_channels=in_channels, 
                     interm_channels=interm_channels, 
                     input_scale=input_scale, 
                     n_layers=n_layers, 
                     filter_type=filter_type, 
                     filter_options=filter_options,
                     norm_type=norm_type,
                     init=init)
            self.n_layers = n_layers
        else:
            self.layers = nn.ModuleList([
                MFNLayer(in_channels=in_channels, 
                        interm_channels=interm_channels, 
                        input_scale=input_scale, 
                        n_layers=n_layers, 
                        filter_type=filter_type,
                        filter_options=filter_options,
                        norm_type=norm_type,
                        init=init) for i in range(n_layers)
            ])

        self.interm_channels = interm_channels
        self.filter_type = filter_type
        self.init_filter = get_filter(filter_type, in_channels, interm_channels, input_scale, n_layers, init)

        init_filter_in = torch.ones(1, self.interm_channels) if self.filter_type in ['fourier', 'gabor'] else torch.zeros(1, self.interm_channels)
        self.register_buffer('init_filter_in', init_filter_in)

    def forward(self, x, z, used_cached=False):
        if self.init_filter is not None:
            gx = self.init_filter(x, self.init_filter_in, used_cached)
        
            z = z + gx

        if self.weight_shared:
            for i in range(self.n_layers):
                z = self.shared_layer(x, z, used_cached)
        else:
            for i, l in enumerate(self.layers):
                z = l(x, z, used_cached)
            
        return z