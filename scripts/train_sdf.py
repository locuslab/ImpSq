import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import numpy as np
import configargparse
import skvideo.io
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pickle

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from modules.modeling_utils import get_model, construct_model_args
from modules.utils import batch_indices_generator, batched_apply, create_mesh
from modules.dataset import PointCloud

def sdf_loss(model_output, gt):
    '''
    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''

    gradient = model_output['grad']
    pred_sdf = model_output['output']

    gt_sdf = gt['sdf'].to(pred_sdf.device)
    gt_normals = gt['normals'].to(gradient.device)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf).to(pred_sdf.device))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf).to(pred_sdf.device), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]).to(gradient.device))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig

def train(args, train_dataloader, model, opt=None, iters=10000, device='cuda', liveplot=False, run_label=None):
    """Standard training/evaluation epoch over the dataset"""
    criterion = lambda model_outputs, gt: sdf_loss(model_outputs, gt)

    data_iter = tqdm(train_dataloader)
    if run_label is not None:
        data_iter.set_description(run_label)

    log_writer = SummaryWriter(args.log_dir)
    
    step_list = []
    loss_list = []
    step_time_list = []
    test_psnr_list = []

    model = model.to(device)

    # Main training Loop
    postfix = {'loss': np.inf, 'psnr': 0., 'forward_steps': 0}
    for i, (model_in, gt) in enumerate(data_iter):
        start_time = time.time()
        
        z_init = torch.zeros(1, model.interm_channels).to(device)
        
        model_outputs = model(model_in.squeeze().to(device), z_init, skip_solver=False, verbose=False, include_grad=True)

        loss_dict = criterion(model_outputs, gt)
        
        loss = 0.
        for loss_name, single_loss in loss_dict.items():
            log_writer.add_scalar(loss_name, single_loss.item(), i + 1)
            loss += single_loss
        loss_list.append(loss.item())
        log_writer.add_scalar('loss', loss.item(), i + 1)

        if opt:
            opt.zero_grad()
            loss.backward()

            opt.step()
        
        postfix['forward_steps'] = model_outputs['forward_steps']
        postfix['loss'] = loss.item()
        
        step_time = time.time() - start_time
        step_time_list.append(step_time)

        do_log = (i + 1) % args.log_freq == 0
        do_vis = (i + 1) % args.vis_freq == 0

        if (i + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), f'{args.save_dir:s}/model_step{i:d}.pth')

        if do_log:
            w_arr = np.linspace(-1, 1, 512, dtype=np.float32)
            slice_coords_2d = torch.tensor(np.stack(np.meshgrid(w_arr, w_arr, indexing='ij'), axis=-1), dtype=torch.float32).view(-1, 2)

            with torch.no_grad():
                yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1).to(device)
                # yz_model_out = model(yz_slice_coords)
                yz_model_out = batched_apply(model, yz_slice_coords, batch_size=128 * 128, print_progress=False)
                sdf_values = yz_model_out.unsqueeze(0)
                sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_contour_plot(sdf_values)
                log_writer.add_figure('yz_sdf_slice', fig, global_step=i + 1)

                del yz_slice_coords, yz_model_out

                xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                            torch.zeros_like(slice_coords_2d[:, :1]),
                                            slice_coords_2d[:,-1:]), dim=-1).to(device)

                xz_model_out = batched_apply(model, xz_slice_coords, batch_size=128 * 128, print_progress=False)
                sdf_values = xz_model_out.unsqueeze(0)
                sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_contour_plot(sdf_values)
                log_writer.add_figure('xz_sdf_slice', fig, global_step=i + 1)

                del xz_slice_coords, xz_model_out

                xy_slice_coords = torch.cat((slice_coords_2d,
                                            -0.75*torch.ones_like(slice_coords_2d[:, :1])), dim=-1).to(device)

                xy_model_out = batched_apply(model, xy_slice_coords, batch_size=128 * 128, print_progress=False)
                sdf_values = xy_model_out.unsqueeze(0)
                sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
                fig = make_contour_plot(sdf_values)
                log_writer.add_figure('xy_sdf_slice', fig, global_step=i + 1)

                del xy_slice_coords, xy_model_out
        
        if do_vis:
            create_mesh(model, args.vis_dir + '/model_step{:d}'.format(i + 1), device=args.device)

        data_iter.set_postfix(postfix)
    
    log_writer.close()

    create_mesh(model.to(args.device), args.vis_dir + '/final', N=512, device=args.device)

    return {
        'loss': loss_list,
        'step_time': step_time_list,
        'step': step_list,
        'test_psnr': test_psnr_list,
        }

def test(args, model):
    assert args.restore_path is not None, 'Restore path cannot be empty'

    create_mesh(model.to(args.device), args.vis_dir + '/test', N=512, device=args.device)

def main(args):

    model_args = construct_model_args(
        model_type=args.model_type,
        n_layers=args.n_layers, 
        in_channels=3,
        interm_channels=args.interm_channels, 
        out_channels=1,
        input_scale=args.input_scale,
        use_implicit=args.use_implicit, 
        filter_type=args.filter_type,
        filter_options={'alpha': args.gabor_alpha},
        norm_type=args.norm_type,
        forward_solver=args.forward_solver,
        backward_solver=args.backward_solver,
    )
    model = get_model(model_args)
    print(model)

    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.inference:
        _point_cloud_paths = {
            'thai_statue': './data/thai_statue.xyz',    
            'room': './data/interior_room.xyz'
        }

        dataset = PointCloud(_point_cloud_paths[args.dataset], on_surface_points=args.train_batch_size, max_iters=args.max_train_iters)
        dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=1, shuffle=False)
        train(
            args,
            dataloader,
            model,
            opt,
            iters=args.max_train_iters,
            device=args.device
        )
    else:
        test(
            args,
            model
        )

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--experiment_id', default='sdf', type=str)
    parser.add_argument('--dataset', default='thai_statue', choices=['thai_statue', 'room'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inference', default=False, action='store_true')

    parser.add_argument('--restore_path', default=None, type=str)

    parser.add_argument('--vis_freq', default=2000, type=int)
    parser.add_argument('--log_freq', default=2000, type=int)
    parser.add_argument('--save_freq', default=2000, type=int)

    parser.add_argument('--max_train_iters', default=10000, type=int)

    parser.add_argument('--model_type', default='implicit', choices=['implicit', 'siren', 'ffn'])
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--interm_channels', default=256, type=int)
    parser.add_argument('--use_implicit', default=False, action='store_true')
    parser.add_argument('--input_scale', default=256., type=float)
    parser.add_argument('--filter_type', default='fourier', choices=['fourier', 'gabor', 'siren_like'])
    parser.add_argument('--norm_type', default='none', choices=['none', 'spectral_norm', 'weight_norm'])
    parser.add_argument('--gabor_alpha', default=6., type=float)
    parser.add_argument('--forward_solver', default='forward_iter', type=str)
    parser.add_argument('--backward_solver', default='forward_iter', type=str)

    parser.add_argument('--train_batch_size', default=50000, type=int)
    parser.add_argument('--test_batch_size', default=50000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    args = parser.parse_args()

    args.log_dir = f'logs/sdf/{args.experiment_id}'
    args.vis_dir = f'{args.log_dir}/visualizations'
    args.save_dir = f'{args.log_dir}/saved_models'

    [os.makedirs(path, exist_ok=True) for path in (args.log_dir, args.vis_dir, args.save_dir)]
    main(args)