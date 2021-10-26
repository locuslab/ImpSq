import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from modules.solvers import *
from modules.structs import MFN
from modules.utils import *

class DEQImplicitLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 interm_channels, 
                 input_scale=256., 
                 filter_type='fourier', 
                 filter_options={},
                 norm_type='none',
                 n_layers=1, 
                 tol=1e-2, 
                 max_iter=100, 
                 direct_forward=False, 
                 one_pass=False, 
                 forward_solver='forward_iter', 
                 backward_solver='forward_iter',
                 init='default',
                 **kwargs):
        super().__init__()
        self.f = MFN(in_channels, interm_channels, n_layers, input_scale=input_scale, filter_type=filter_type, norm_type=norm_type, filter_options=filter_options, init=init)
        self.in_channels = in_channels
        self.interm_channels = interm_channels
        
        _forward_solvers = {
            'broyden': broyden,
            'anderson': anderson,
            'forward_iter': forward_iteration
        }
        _backward_solvers = {
            'broyden': BroydenBackward,
            'anderson': AndersonBackward,
            'forward_iter': ForwardIterBackward,
            'onestep': OnestepBackward,
        }
        self.forward_solver = _forward_solvers[forward_solver]
        self.backward_solver = _backward_solvers[backward_solver]
        self.prev_g = None
        
        self.warmup_iters = 0
        self.tol = tol
        self.direct_forward = direct_forward
        self.one_pass = one_pass
        self.max_iter = max_iter
    
    def forward(self, x, z, skip_solver=False, verbose=False):
        if self.one_pass:
            # Run a one pass model instead of solving equilibrium point
            f = self.f(x, z)
            nstep = 0
            lowest = 0.
        else:

            # Use IFT and inverse jacobian approximation to compute gradients
            if not skip_solver:
                with torch.no_grad():
                    solver_stats = self.forward_solver(lambda _z, use_cached=False: self.f(x, _z, use_cached), z, max_iter=self.max_iter, tol=self.tol, stop_mode='rel')
                    z = solver_stats['result']
                    nstep = solver_stats['nstep']
                    lowest = solver_stats['lowest']
            else:
                nstep = 0
                lowest = 0.
            z = z.requires_grad_()
            f = self.f(x, z)
            
            if verbose:
                print("Forward: Iterations {:d} Error {:e}".format(nstep, lowest))

            f = self.backward_solver.apply(f, z)
        return {
            'output': f,
            'forward_steps': nstep
        }

class DEQImplicitNN(nn.Module):
    def __init__(self, implicit_layer, output_channels, output_linear=True):
        super(DEQImplicitNN, self).__init__()
        self.add_module('implicit_layer', implicit_layer)
        self.interm_channels = self.implicit_layer.interm_channels
        self.output_map = nn.Linear(self.interm_channels, output_channels)
        self.output_linear = output_linear

    def forward(self, x, z=None, skip_solver=False, verbose=False, include_grad=False):
        b = x.size(0)
        if z is None:
            z = torch.zeros(*x.shape[:-1], self.interm_channels).to(x.device)

        if include_grad:
            x.requires_grad_(True)
        imp_layer_output = self.implicit_layer(x, z, skip_solver=skip_solver, verbose=verbose)

        output = self.output_map(imp_layer_output['output'])
        if not self.output_linear:
            output = torch.sin(output)

        ret_dict = {
            'output': output,
            'imp_layer_output': imp_layer_output['output'],
            'forward_steps': imp_layer_output['forward_steps']
        }

        if include_grad:
            grad = torch.autograd.grad(output, [x], grad_outputs=torch.ones_like(output).to(output.device), create_graph=True)[0]
            ret_dict['grad'] = grad

        return ret_dict
        
    def decode(self, z):
        if self.z_map is not None:
            z = self.z_map(z)
        return self.output_map(z)

class SIREN(nn.Module):
    def __init__(self, in_channels, interm_channels, out_channels, n_layers=4, scale=100, omega=30., output_linear=True):
        super().__init__()
        _linear_maps = \
            ([nn.Linear(in_channels, interm_channels)]
            + [nn.Linear(interm_channels, interm_channels) for i in range(n_layers - 1)]
            + [nn.Linear(interm_channels, out_channels)])
        
        print(scale)
        self.omega=omega
        self.linear_maps = nn.ModuleList(_linear_maps)
        # self.linear_maps[0].weight.data *= scale
        self.scale = scale
        nn.init.uniform_(self.linear_maps[0].weight, -1 / in_channels, 1 / in_channels)
        for i in range(1, len(self.linear_maps)):
            nn.init.uniform_(
                self.linear_maps[i].weight,
                -np.sqrt(6 / interm_channels) / self.omega,
                np.sqrt(6 / interm_channels) / self.omega
            )
        self.output_linear = output_linear
        self.interm_channels = interm_channels

    def forward(self, x, z=None, **kwargs):
        h = x
        for i in range(len(self.linear_maps) - 1):
            if i > 0:
                h = torch.sin(self.omega * self.linear_maps[i](h))
            else:
                h = torch.sin(self.scale * self.linear_maps[i](h))
        h = self.linear_maps[-1](h)
        if not self.output_linear:
            h = torch.sin(self.omega * h)
        return {
            'output': h
        }

class FFN(nn.Module):
    def __init__(self, in_channels, interm_channels, out_channels, n_layers=4, output_linear=False):
        super().__init__()
        self.embedder = GaussianFFNEmbedder(in_channels)
        _linear_maps = \
            ([nn.Linear(self.embedder.out_channels, interm_channels)]
            + [nn.Linear(interm_channels, interm_channels) for i in range(n_layers - 1)]
            + [nn.Linear(interm_channels, out_channels)])
        
        self.linear_maps = nn.ModuleList(_linear_maps)
        self.output_linear = output_linear
        self.interm_channels = interm_channels
    
    def forward(self, x, z=None, **kwargs):
        h = self.embedder(x)
        for m in self.linear_maps[:-1]:
            h = torch.relu(m(h))
        if not self.output_linear:
            h = torch.sin(self.linear_maps[-1](h))
        return {
            'output': h
        }
