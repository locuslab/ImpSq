import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import livelossplot
import skimage
import configargparse

from modules.modeling_utils import get_model, construct_model_args, get_summary_dict
from modules.utils import get_psnr

def load_dataset(filename, id):
    npz_data = np.load(filename)
    out = {
        "data_train":npz_data['train_data'] / 255.,
        "data_test":npz_data['test_data'] / 255.,
    }
    return out

def preprocess(img_tensor):
    return img_tensor * 2 - 1

def postprocess(img_tensor):
    return torch.clamp(((img_tensor + 1) / 2), 0, 1).squeeze(-1).detach().cpu().numpy()

def hw2batch(hw_tensor):
    b, h, w, c = hw_tensor.shape
    batched = hw_tensor.reshape(-1, c)
    return batched, h, w

def batch2hw(batched_tensor, h, w):
    c = batched_tensor.size(1)
    hw = batched_tensor.view(-1, h, w, c).contiguous()
    return hw

def train(args, train_data, test_data, model, opt=None, scheduler=None, iters=5000, device='cuda', visualize=False, vis_tag=None, vis_freq=100, log_freq=50, use_cached_input=False, run_label=None):
    """Standard training/evaluation epoch over the dataset"""
    train_in, train_tgt = train_data
    test_in, test_tgt = test_data
    criterion = lambda x, y: torch.mean((x - y) ** 2)

    train_batched_input, train_h, train_w = hw2batch(train_in)
    train_batched_target, _, _ = hw2batch(preprocess(train_tgt))

    test_batched_input, test_h, test_w = hw2batch(test_in)
    test_batched_target, _, _ = hw2batch(preprocess(test_tgt))

    data_iter = tqdm(range(iters), leave=True)
    if run_label is not None:
        data_iter.set_description(run_label)

    z_init = torch.zeros(1, model.interm_channels).to(device)
    
    step_list = []
    loss_list = []
    step_time_list = []
    train_psnr_list = []
    test_psnr_list = []
    forward_steps = []

    model = model.to(device)
    max_mem = 0

    # Main training Loop
    for i in data_iter:
        start_time = time.time()

        model_outputs = model(train_batched_input, z_init, skip_solver=False, verbose=False)
        forward_steps.append(model_outputs['forward_steps'])
        data_iter.set_postfix({'forward step': model_outputs['forward_steps']})

        if use_cached_input:
            z_init = model_outputs['imp_layer_output'].detach()

        loss = criterion(model_outputs['output'], train_batched_target)
        loss_list.append(loss.item())

        max_mem = max(max_mem, torch.cuda.memory_allocated(device))

        if opt:
            opt.zero_grad()
            loss.backward()

            opt.step()

            if scheduler:
                scheduler.step()

        step_time = time.time() - start_time
        step_time_list.append(step_time)

        do_visualize = (visualize and (i + 1) % vis_freq == 0)
        do_log = ((i + 1) % log_freq == 0)

        if do_log or do_visualize:
            with torch.no_grad():
                train_orig = postprocess(batch2hw(train_batched_target, train_h, train_w))
                train_rec = postprocess(batch2hw(model(train_batched_input, torch.zeros(1, model.interm_channels).to(device))['output'], train_h, train_w))

                test_orig = postprocess(batch2hw(test_batched_target, test_h, test_w))
                test_rec = postprocess(batch2hw(model(test_batched_input, torch.zeros(1, model.interm_channels).to(device))['output'], test_h, test_w))
            
            if do_log:
                step_list.append(i + 1)

                train_psnr = get_psnr(train_orig, train_rec)
                train_psnr_list.append(train_psnr)
                    
                test_psnr = get_psnr(test_orig, test_rec)
                test_psnr_list.append(test_psnr)
                data_iter.set_postfix({'Train PSNR': train_psnr, 'Test PSNR': test_psnr})

            if do_visualize:
                fig, ax = plt.subplots(1, 4)
                fig.set_size_inches(16, 4)

                ax[0].clear()
                ax[0].set_axis_off()
                ax[0].imshow(train_orig[0], cmap='gray')
                ax[1].clear()
                ax[1].set_axis_off()
                ax[1].imshow(train_rec[0], cmap='gray')
                ax[2].clear()
                ax[2].set_axis_off()
                ax[2].imshow(test_orig[0], cmap='gray')
                ax[3].clear()
                ax[3].set_axis_off()
                ax[3].imshow(test_rec[0], cmap='gray')
                if vis_tag is None:
                    plt.show()
                    plt.close()
                else:
                    fig.savefig('{:s}/step_{:d}.jpeg'.format(args.vis_dir, i + 1), dpi=600)
    return {
        'loss': loss_list,
        'step_time': step_time_list,
        'step': step_list,
        'train_psnr': train_psnr_list,
        'test_psnr': test_psnr_list,
        'max_mem': max_mem,
        'forward_steps': forward_steps,
        }

def main(args):
    if args.dataset == 'nature':
        # import div2k
        dataset = load_dataset('./data/data_div2k.npz', '1TtwlEDArhOMoH18aUyjIMSZ3WODFmUab')
        data_channels = 3
    elif args.dataset == 'text':
        # import text
        dataset = load_dataset('./data/data_2d_text.npz', '1V-RQJcMuk9GD4JCUn70o7nwQE0hEzHoT')
        data_channels = 3
    elif args.dataset == 'camera':
        # import cameraman
        camera_image = skimage.data.camera()
        dataset = {
            'data_test' : (camera_image.reshape(1, 512, 512, 1).astype(np.float32) / 255)
        }
        data_channels = 1
        

    RES = 512

    full_x = np.linspace(0, 1, RES) * 2 - 1
    full_x_grid = torch.tensor(np.stack(np.meshgrid(full_x,full_x), axis=-1)[None, :, :], dtype=torch.float32)

    if args.dataset in ['nature', 'text']:
        x_train = full_x_grid[:, ::2, ::2]
        x_test = full_x_grid[:, 1::2, 1::2]

        y_train = dataset['data_test'][:, ::2, ::2]
        y_test = dataset['data_test'][:, 1::2, 1::2]
    else:
        x_train = x_test = full_x_grid#[:, ::2, ::2]
        y_train = y_test = dataset['data_test']#[:, ::2, ::2]

    x_train, x_test = x_train.to(args.device), x_test.to(args.device)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(args.device), torch.tensor(y_test, dtype=torch.float32).to(args.device)

    # model_args = construct_model_args(
    #     model_type='implicit',
    #     n_layers=1, 
    #     in_channels=2,
    #     interm_channels=512, 
    #     out_channels=3,
    #     input_scale=256,
    #     use_implicit=True, 
    #     filter_type='gabor',
    #     forward_solver='forward_iter',
    #     backward_solver='onestep',
    # )
    # model = get_model(model_args)
    # print(model)

    # opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # train_stats = train(
    #         args,
    #         (x_train, y_train[0].unsqueeze(0)),
    #         (x_test, y_test[0].unsqueeze(0)),
    #         model,
    #         opt,
    #         log_freq=args.log_freq,
    #         iters=args.max_train_iters,
    #         device=args.device,
    #         use_cached_input=False,
    # )

    # train_steps_no_cache = train_stats['forward_steps'],
    # print(train_steps_no_cache)

    # model = get_model(model_args)
    # print(model)

    # opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # train_stats = train(
    #     args,
    #     (x_train, y_train[0].unsqueeze(0)),
    #     (x_test, y_test[0].unsqueeze(0)),
    #     model,
    #     opt,
    #     log_freq=args.log_freq,
    #     iters=args.max_train_iters,
    #     device=args.device,
    #     use_cached_input=True,
    # )

    # train_steps_cached = train_stats['forward_steps']
    # print(train_steps_cached)

    # np.savez('misc/forward_iters.npz', 
    #     cached=np.array(train_steps_cached),
    #     mo_cache=np.array(train_steps_no_cache)
    # )

    torch.random.manual_seed(0)

    model_args = construct_model_args(
        model_type='implicit',
        n_layers=1, 
        in_channels=2,
        interm_channels=256, 
        out_channels=3,
        input_scale=128,
        use_implicit=True, 
        filter_type='gabor',
        forward_solver='forward_iter',
        backward_solver='fpn',
    )
    model = get_model(model_args)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    train_stats = train(
            args,
            (x_train, y_train[0].unsqueeze(0)),
            (x_test, y_test[0].unsqueeze(0)),
            model,
            opt,
            log_freq=args.log_freq,
            iters=args.max_train_iters,
            device=args.device,
            use_cached_input=True,
    )

    err_fpn = train_stats['loss'],
    print(err_fpn)

    torch.random.manual_seed(0)

    model_args = construct_model_args(
        model_type='implicit',
        n_layers=1, 
        in_channels=2,
        interm_channels=256, 
        out_channels=3,
        input_scale=128,
        use_implicit=True, 
        filter_type='gabor',
        forward_solver='forward_iter',
        backward_solver='onestep',
    )
    model = get_model(model_args)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    train_stats = train(
            args,
            (x_train, y_train[0].unsqueeze(0)),
            (x_test, y_test[0].unsqueeze(0)),
            model,
            opt,
            log_freq=args.log_freq,
            iters=args.max_train_iters,
            device=args.device,
            use_cached_input=True,
    )

    err_onestep = train_stats['loss'],
    print(err_onestep)

    torch.random.manual_seed(0)

    model_args = construct_model_args(
        model_type='implicit',
        n_layers=1, 
        in_channels=2,
        interm_channels=256, 
        out_channels=3,
        input_scale=128,
        use_implicit=True, 
        filter_type='gabor',
        forward_solver='forward_iter',
        backward_solver='forward_iter',
    )
    model = get_model(model_args)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    train_stats = train(
        args,
        (x_train, y_train[0].unsqueeze(0)),
        (x_test, y_test[0].unsqueeze(0)),
        model,
        opt,
        log_freq=args.log_freq,
        iters=args.max_train_iters,
        device=args.device,
        use_cached_input=True,
    )

    err_full = train_stats['loss']
    print(err_full)

    np.savez('misc/err.npz', 
        fpn=np.array(err_fpn),
        onestep=np.array(err_onestep),
        full=np.array(err_full)
    )

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--experiment_id', default='vanilla')
    parser.add_argument('--dataset', default='nature', choices=['camera', 'nature', 'text'])
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=1., type=float)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--input_scale', default=256., type=float)

    parser.add_argument('--max_train_iters', default=1000, type=int)
    parser.add_argument('--log_freq', default=50, type=int)

    parser.add_argument('--continue_run', default=False, action='store_true')

    parser.add_argument('--inference', default=False, action='store_true')

    args = parser.parse_args()

    args.log_dir = f'logs/images/{args.experiment_id}'
    main(args)