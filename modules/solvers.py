import torch
import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
from torch import autograd

import numpy as np 
import pickle
import sys
import os
from scipy.optimize import root
import time

def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.
    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite

def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, L')
    # part_Us: (N, L', threshold)
    # part_VTs: (N, threshold, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bi, bid -> bd', x, part_Us)   # (N, threshold)
    return -x + torch.einsum('bd, bdi -> bi', xTU, part_VTs)    # (N, L')


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, L')
    # part_Us: (N, L', threshold)
    # part_VTs: (N, threshold, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdi, bi -> bd', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bid, bd -> bi', part_Us, VTx)     # (N, L')

########################################################################
# Solvers
########################################################################

def forward_iteration(f, x0, max_iter=100, tol=1e-2, stop_mode='rel'):
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    for k in range(1, max_iter + 1):
        if k == 1:
            x = x0
        else:
            x = f0
        f0 = f(x, use_cached=k > 1)
        lowest_xest = f0
        
        abs_diff = torch.norm(f0 - x).item()
        rel_diff = abs_diff / (torch.norm(f0).item() + 1e-9)
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k
                if mode == stop_mode and lowest_dict[mode] < tol: 
                    lowest_xest = f0

        if trace_dict[stop_mode][-1] < tol:
            for _ in range(max_iter-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {"result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict['abs'],
        "rel_trace": trace_dict['rel'],
        "tol": tol,
        "max_iter": max_iter}
    return out

def broyden(f, x0, max_iter=100, tol=1e-2, stop_mode="rel", ls=False):
    bsz, seq_len = x0.size()
    g = lambda y: f(y) - y
    dev = x0.device
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    
    x_est = x0           # (bsz, L')
    gx = g(x_est)        # (bsz, L')
    nstep = 0
    tnstep = 0

    m = 5
    
    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, seq_len, m).to(dev)     # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, m, seq_len).to(dev)
    update = -matvec(Us[:,:,:nstep], VTs[:,:nstep], gx)      # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False
    
    # To be used in protective breaks
    protect_thres = 1e3 * seq_len
    new_objective = 1e8

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    for nstep in range(1, max_iter + 1):
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        tnstep += (ite+1)
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode: 
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        new_objective = diff_dict[stop_mode]
        if new_objective < tol: break
        if new_objective < 3*tol and nstep > 30 and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        n = min(nstep - 1, m)

        part_Us, part_VTs = Us[:,:,:n], VTs[:,:n]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bi, bi -> b', vT, delta_gx)[:,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % m] = vT
        Us[:,:,(nstep-1) % m] = u
        update = -matvec(Us[:,:,:n + 1], VTs[:,:n + 1], gx)

    # Fill everything up to the threshold length (even if )
    for _ in range(max_iter+1-len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return {"result": lowest_xest,
            "lowest": lowest_dict[stop_mode],
            "nstep": lowest_step_dict[stop_mode],
            "prot_break": prot_break,
            "abs_trace": trace_dict['abs'],
            "rel_trace": trace_dict['rel'],
            "tol": tol,
            "max_iter": max_iter}

def anderson(f, x0, max_iter=100, tol=1e-2, m=5, lam=1e-4, stop_mode='rel', beta=1.0, **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, L, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].reshape_as(x0)).reshape(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}

    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:,k%m] - X[:,k%m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:,k%m].norm().item())
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode: 
                    lowest_xest =  X[:,k%m].view_as(x0).clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < tol:
            for _ in range(max_iter-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {"result": lowest_xest,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "tol": tol,
           "max_iter": max_iter}
    X = F = None
    return out

class DEQBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, z):
        ctx.save_for_backward(f, z)
        return f

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError

class OnestepBackward(DEQBackward):
    @staticmethod
    def backward(ctx, grad):
        f, z = ctx.saved_tensors
        return autograd.grad(f, z, grad, retain_graph=True)[0] + grad, None

class TwostepBackward(DEQBackward):
    @staticmethod
    def backward(ctx, grad):
        f, z = ctx.saved_tensors

        y = grad
        for i in range(2):
            y = autograd.grad(f, z, y, retain_graph=True)[0] + grad
        return y, None

class ForwardIterBackward(DEQBackward):
    @staticmethod
    def backward(ctx, grad):
        f, z = ctx.saved_tensors

        tol = 1e-2
        solver_stats = forward_iteration(lambda y, use_cached=False : autograd.grad(f, z, y, retain_graph=True)[0] + grad,
                                            grad,
                                            max_iter=100,
                                            tol=tol,
                                            stop_mode='rel')

        backward = solver_stats['result']
        backward_steps = solver_stats['nstep']

        # print("Backward: Iterations {:d}, Error {:e}".format(backward_steps, backward_res))
        return backward, None

class AndersonBackward(DEQBackward):
    @staticmethod
    def backward(ctx, grad):
        f, z = ctx.saved_tensors

        tol = 1e-2
        solver_stats = anderson(lambda y : autograd.grad(f, z, y, retain_graph=True)[0] + grad,
                                        grad,
                                        max_iter=100,
                                        tol=tol,
                                        stop_mode='rel')
        backward = solver_stats['result']
        backward_res = solver_stats['lowest']
        backward_steps = solver_stats['nstep']

        # print("Backward: Iterations {:d}, Error {:e}".format(backward_steps, backward_res))
        return backward, None

class BroydenBackward(DEQBackward):
    @staticmethod
    def backward(ctx, grad):
        f, z = ctx.saved_tensors

        tol = 1e-2
        solver_stats = broyden(lambda y : autograd.grad(f, z, y, retain_graph=True)[0] + grad,
                                        grad,
                                        max_iter=100,
                                        tol=tol,
                                        stop_mode='rel')

        backward = solver_stats['result']
        backward_res = solver_stats['lowest']
        backward_steps = solver_stats['nstep']

        if backward_res > tol:
            print(solver_stats['abs_trace'])
            print(solver_stats['rel_trace'])
            eye_mat = torch.eye(z.shape[-1]).to(z.device)

            # input("Press enter to save grad and jacobian for debugging")

            # n_samples = 5000
            # indices = np.random.choice(f.shape[0], n_samples, replace=False)
            # grad_mat = torch.zeros(n_samples, z.shape[-1], f.shape[-1])
            # for j in range(z.shape[-1]):
            #     grad_mat[:, j] = autograd.grad(f, z, grad_outputs=eye_mat[j].unsqueeze(0).expand_as(z), retain_graph=True)[0][indices]
            # np.save('./cache/cached_jac.npy', grad_mat.cpu().numpy())
            # np.save('./cache/cached_grad.npy', grad[indices].cpu().numpy())

            raise RuntimeError()

        # print("Backward: Iterations {:d}, Error {:e}".format(backward_steps, backward_res))
        return backward, None
