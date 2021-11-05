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
from modules.utils import batch_indices_generator, batched_apply, ConfusionMatrix

import skimage.io

import trimesh
import pyembree

_mesh_paths = {
    'dragon': './data/dragon.ply',
    'bunny': './data/bunny.ply',
    'buddha': './data/buddha.ply',
    'armadillo': './data/armadillo.ply',
    'lucy': './data/lucy.ply'
}

gt_fn = lambda queries, mesh : mesh.ray.contains_points(queries.reshape([-1,3])).reshape(queries.shape[:-1])

def make_test_pts(mesh, corners, test_size=2**20):
  c0, c1 = corners
  test_easy = np.random.uniform(size=[test_size, 3]) * (c1-c0) + c0
  batch_pts, batch_normals = get_normal_batch(mesh, test_size)
  test_hard = batch_pts + np.random.normal(size=[test_size,3]) * .001
  return test_easy, test_hard

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def recenter_mesh(mesh):
  mesh.vertices -= mesh.vertices.mean(0)
  mesh.vertices /= np.max(np.abs(mesh.vertices))
  mesh.vertices = .5 * (mesh.vertices + 1.)


def load_mesh(mesh_name, verbose=True):

  mesh = trimesh.load(_mesh_paths[mesh_name])
  mesh = as_mesh(mesh)
  if verbose: 
    print(mesh.vertices.shape)
  recenter_mesh(mesh)

  c0, c1 = mesh.vertices.min(0) - 1e-3, mesh.vertices.max(0) + 1e-3
  corners = [c0, c1]
  if verbose:
    print(c0, c1)
    print(c1-c0)
    print(np.prod(c1-c0))
    print(.5 * (c0+c1) * 2 - 1)

  return mesh, corners

def load_test_pts(mesh_name, mesh_obj=None, regen=True, verbose=True):
    test_pt_file = os.path.join(os.path.split(_mesh_paths[mesh_name])[0], mesh_name + '_test_pts.npz')
    if mesh_obj is None:
        mesh, corners = load_mesh(mesh_name)
    else:
        mesh, corners = mesh_obj
    if regen or not os.path.exists(test_pt_file):
        test_pts_easy, test_pts_hard  = make_test_pts(mesh, corners)
        np.savez(test_pt_file, easy=test_pts_easy, hard=test_pts_hard)
    else:
        if verbose: print('load pts')
        test_pts_dict = np.load(test_pt_file)
        test_pts_easy, test_pts_hard = test_pts_dict['easy'], test_pts_dict['hard']

    if verbose: print(test_pts_easy.shape)

    test_labels_easy = gt_fn(test_pts_easy, mesh)
    test_labels_hard = gt_fn(test_pts_hard, mesh)
    if verbose:
        print(f"Test points [easy] - Inside obj: {np.sum(test_labels_easy):d} - Outside obj: {np.sum(1 - test_labels_easy):d}")
        print(f"Test points [hard] - Inside obj: {np.sum(test_labels_hard):d} - Outside obj: {np.sum(1 - test_labels_hard):d}")
    return {
        'easy': (test_pts_easy, test_labels_easy), 
        'hard': (test_pts_hard, test_labels_hard)
    }

###################



trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], rays_d.shape)
    return np.stack([rays_o, rays_d], 0)

#########


def render_rays_native_hier(model, rays, corners, near, far, N_samples, N_samples_2, clip, device): #, rand=False):
    rays_o, rays_d = rays
    c0, c1 = corners

    th = .5
    
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts = 0.5 * (pts + 1)

    h, w, d = pts.shape[:-1]
    
    # Run network
    model_output = batched_apply(model, pts.view(-1, 3), batch_size=50000, device=device)
    alpha = torch.sigmoid(model_output).view(h, w, d)
    if clip:
      mask = torch.logical_or(torch.any(pts < c0, -1), torch.any(pts > c1, -1)).to(device)
      alpha = torch.where(mask, torch.zeros_like(alpha).to(device), alpha)

    alpha = torch.where(alpha > th, torch.ones_like(alpha).to(device), torch.zeros_like(alpha).to(device))

    trans = 1.-alpha + 1e-10
    trans = torch.cat([torch.ones_like(trans[...,:1]).to(trans.device), trans[...,:-1]], -1)  
    weights = alpha * torch.cumprod(trans, -1)
    
    depth_map = torch.sum(weights * z_vals.to(device), -1)
    acc_map = torch.sum(weights, -1)

    # Second pass to refine isosurface

    z_vals = torch.linspace(-1., 1., N_samples_2) * .01 + depth_map[...,None].cpu()
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts = 0.5 * (pts + 1)

    # Run network
    model_output = batched_apply(model, pts.view(-1, 3), batch_size=50000, device=device)
    alpha = torch.sigmoid(model_output).view(h, w, d)
    if clip:
      mask = torch.logical_or(torch.any(pts < c0, -1), torch.any(pts > c1, -1)).to(device)
      alpha = torch.where(mask, torch.zeros_like(alpha).to(device), alpha)

    alpha = torch.where(alpha > th, torch.ones_like(alpha).to(device), torch.zeros_like(alpha).to(device))

    trans = 1.-alpha + 1e-10
    trans = torch.cat([torch.ones_like(trans[...,:1]).to(trans.device), trans[...,:-1]], -1)  
    weights = alpha * torch.cumprod(trans, -1)
    
    depth_map = torch.sum(weights * z_vals.to(device), -1)
    acc_map = torch.sum(weights, -1)

    return depth_map, acc_map

def make_normals(rays, depth_map):
  rays_o, rays_d = rays
  pts = rays_o + rays_d * depth_map[...,None]
  dx = pts - torch.roll(pts, -1, dims=0)
  dy = pts - torch.roll(pts, -1, dims=1)
  normal_map = torch.cross(dx, dy)
  normal_map = normal_map / torch.clamp_min(torch.norm(normal_map, dim=-1, keepdim=True), 1e-5)
  return normal_map


def render_mesh_normals(mesh, rays):
  origins, dirs = rays.reshape([2,-1,3])
  origins = origins * .5 + .5
  dirs = dirs * .5
  z = mesh.ray.intersects_first(origins, dirs)
  pic = np.zeros([origins.shape[0],3]) 
  pic[z!=-1] = mesh.face_normals[z[z!=-1]]
  pic = np.reshape(pic, rays.shape[1:])
  return pic

def uniform_bary(u):
  su0 = np.sqrt(u[..., 0])
  b0 = 1. - su0
  b1 = u[..., 1] * su0
  return np.stack([b0, b1, 1. - b0 - b1], -1)


def get_normal_batch(mesh, bsize):

  batch_face_inds = np.array(np.random.randint(0, mesh.faces.shape[0], [bsize]))
  batch_barys = np.array(uniform_bary(np.random.uniform(size=[bsize, 2])))
  batch_faces = mesh.faces[batch_face_inds]
  batch_normals = mesh.face_normals[batch_face_inds]
  batch_pts = np.sum(mesh.vertices[batch_faces] * batch_barys[...,None], 1)

  return batch_pts, batch_normals


gt_fn = lambda queries, mesh : mesh.ray.contains_points(queries.reshape([-1,3])).reshape(queries.shape[:-1])

R = 2.
c2w = pose_spherical(90. + 10 + 45, -30., R)

N_samples = 64
N_samples_2 = 64
H = 256
W = H
focal = H * .9
rays = get_rays(H, W, focal, c2w[:3,:4])

render_args_lr = [get_rays(H, W, focal, c2w[:3,:4]), None, R-1, R+1, N_samples, N_samples_2, True]

N_samples = 256
N_samples_2 = 256
H = 512
W = H
focal = H * .9
rays = get_rays(H, W, focal, c2w[:3,:4])

render_args_hr = [get_rays(H, W, focal, c2w[:3,:4]), None, R-1, R+1, N_samples, N_samples_2, True]


def train(args, mesh_obj, model, opt=None, iters=10000, device='cuda', liveplot=False, run_label=None):
    """Standard training/evaluation epoch over the dataset"""
    criterion = lambda x, z: torch.mean(torch.relu(x) - x * z + torch.log(1 + torch.exp(-torch.abs(x))))

    data_iter = tqdm(range(1, iters + 1))
    if run_label is not None:
        data_iter.set_description(run_label)

    mesh, corners = mesh_obj
    c0, c1 = [torch.tensor(t, dtype=torch.float32) for t in corners]
    render_args_hr[0] = [torch.tensor(t, dtype=torch.float32) for t in render_args_hr[0]]
    render_args_lr[0] = [torch.tensor(t, dtype=torch.float32) for t in render_args_lr[0]]
    render_args_hr[1] = [c0, c1]
    render_args_lr[1] = [c0, c1]

    c1, c0 = c1.to(device), c0.to(device)
    
    step_list = []
    loss_list = []
    step_time_list = []
    test_psnr_list = []

    model = model.to(device)

    # Main training Loop
    postfix = {'loss': np.inf, 'psnr': 0., 'forward_steps': 0}
    for i in data_iter:
        start_time = time.time()
        
        inputs = torch.rand(args.train_batch_size, 3).to(device) * (c1 - c0) + c0
        z_init = torch.zeros(1, model.interm_channels).to(device)

        target = torch.tensor(gt_fn(inputs.cpu().numpy(), mesh), dtype=torch.bool).to(device)
        
        model_outputs = model(inputs, z_init, skip_solver=False, verbose=False, include_grad=False)

        pred = torch.sigmoid(model_outputs['output'].squeeze())
        loss = criterion(model_outputs['output'].squeeze(), target.float())
        loss_list.append(loss.item())

        if opt:
            opt.zero_grad()
            loss.backward()

            opt.step()
        
        postfix['forward_steps'] = model_outputs['forward_steps']
        postfix['loss'] = loss.item()
        
        step_time = time.time() - start_time
        step_time_list.append(step_time)

        do_log = i % args.log_freq == 0
        do_vis = i % args.vis_freq == 0

        if i % args.save_freq == 0:
            torch.save(model.state_dict(), f'{args.save_dir:s}/model_step{i:d}.pth')

        if do_log:
            summary_dict = {
                'train_loss': loss_list,
                'step_time': step_time_list
            }

            with open('{:s}/summary.pkl'.format(args.log_dir), 'wb') as summary_f:
                pickle.dump(summary_dict, summary_f)
        
        if do_vis:
            with torch.no_grad():
                depth_map, acc_map = render_rays_native_hier(model, *render_args_lr, device=device)
                normal_map = make_normals(render_args_lr[0], depth_map.cpu()) * 0.5 + 0.5

            fig, axes = plt.subplots(1, 3)
            fig.set_size_inches(6 * 3, 6)

            for ax in axes:
                ax.clear()
                ax.set_axis_off()

            axes[0].imshow(depth_map.squeeze().cpu().numpy())
            axes[1].imshow(acc_map.squeeze().cpu().numpy())
            axes[2].imshow(normal_map.squeeze().cpu().numpy())

            fig.set_tight_layout(True)
            fig.savefig(args.vis_dir + '/vis_step{:d}.png'.format(i))

        data_iter.set_postfix(postfix)

    with torch.no_grad():
        depth_map, acc_map = render_rays_native_hier(model, *render_args_hr, device=device)
        normal_map = make_normals(render_args_hr[0], depth_map.cpu()) * 0.5 + 0.5

    skimage.io.imsave(args.vis_dir + '/final_rendered.png', normal_map.squeeze().cpu().numpy())

    return {
        'loss': loss_list,
        'step_time': step_time_list,
        'step': step_list,
        'test_psnr': test_psnr_list,
        }

def eval(args, model):
    assert args.restore_path is not None, 'Restore path cannot be empty'

    state_dict = torch.load(args.restore_path)
    model.load_state_dict(state_dict)
    model.to(args.device)

    mesh_obj = load_mesh(args.dataset)
    all_tests = load_test_pts(args.dataset, mesh_obj, regen=False, verbose=True)
    log_file = os.path.join(args.log_dir, "test_output.txt")
    with open(log_file, 'w') as f:
        for test_name in all_tests:
            test_pts, test_labels = [torch.tensor(arr, dtype=torch.float32) for arr in all_tests[test_name]]
            test_ds = torch.utils.data.TensorDataset(test_pts, test_labels)
            test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=10000, drop_last=False, pin_memory=True)

            cm = ConfusionMatrix()
            for (pts, labels) in iter(test_loader):
                pts, labels = pts.to(args.device), labels.to(args.device)
                model_outputs = model(pts, skip_solver=False, verbose=False, include_grad=False)
                pred = model_outputs['output'].squeeze() > 0

                # print(pred, labels)
                cm.update(pred.detach().cpu().numpy(), labels.cpu().numpy())

            f.write(f"Test: {test_name}\n")
            f.write(f"\tAccuracy {cm.get_acc():.5f}\n")
            f.write(f"\tPrecision {cm.get_precision():.5f}\n")
            f.write(f"\tRecall {cm.get_recall():.5f}\n")
            f.write(f"\tIoU {cm.get_iou():.5f}\n")
            f.write("\n")

    mesh, corners = mesh_obj
    c0, c1 = [torch.tensor(t, dtype=torch.float32) for t in corners]
    render_args_hr[0] = [torch.tensor(t, dtype=torch.float32) for t in render_args_hr[0]]
    render_args_lr[0] = [torch.tensor(t, dtype=torch.float32) for t in render_args_lr[0]]
    render_args_hr[1] = [c0, c1]
    render_args_lr[1] = [c0, c1]

    with torch.no_grad():
        depth_map, acc_map = render_rays_native_hier(model, *render_args_hr, device=args.device)
        normal_map = make_normals(render_args_hr[0], depth_map.cpu()) * 0.5 + 0.5

    skimage.io.imsave(args.vis_dir + '/final_rendered.png', normal_map.squeeze().cpu().numpy())


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
        tol=args.test_tol if args.eval else args.train_tol
    )
    model = get_model(model_args)
    print(model)

    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.eval:

        mesh_obj = load_mesh(args.dataset)
        train(
            args,
            mesh_obj,
            model,
            opt,
            iters=args.max_train_iters,
            device=args.device
        )
    else:
        eval(
            args,
            model
        )

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--experiment_id', default='vanilla', type=str)
    parser.add_argument('-c', '--config_file', default=None, is_config_file=True)
    parser.add_argument('--dataset', default='dragon', choices=['dragon', 'bunny', 'buddha', 'armadillo', 'lucy'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval', default=False, action='store_true')
    

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
    parser.add_argument('--gabor_alpha', default=3., type=float)
    parser.add_argument('--forward_solver', default='forward_iter', choices=['forward_iter', 'broyden'])
    parser.add_argument('--backward_solver', default='forward_iter', choices=['onestep', 'forward_iter', 'broyden'])
    parser.add_argument('--train_tol', default=1e-3, type=float)
    parser.add_argument('--test_tol', default=1e-4, type=float)

    parser.add_argument('--train_batch_size', default=50000, type=int)
    parser.add_argument('--test_batch_size', default=50000, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)

    args = parser.parse_args()

    args.log_dir = f'logs/3d_occupancy/{args.experiment_id}{args.dataset}'
    args.vis_dir = f'{args.log_dir}/visualizations'
    args.save_dir = f'{args.log_dir}/saved_models'

    [os.makedirs(path, exist_ok=True) for path in (args.log_dir, args.vis_dir, args.save_dir)]
    main(args)