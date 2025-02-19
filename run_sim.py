from utils import (
    config_parser, setup_envs,
    seed_all, setup_network, get_loader,
    get_dataset_size, collect_stats,
    step_env)
import ray
from time import time
from copy import copy
import torch
from tensorboardX import SummaryWriter
from filelock import FileLock
import pickle
import os

from environment import new_env_utils

def optimize(value_net_key, value_net, optimizer, loader,
             criterion, writer, num_updates):
    if loader is None or optimizer is None:
        return
    device = value_net.device
    for _, (obs, action_mask, label) in zip(range(num_updates), loader):
        obs, kp_stack, action_prev, fling_prev, scale = obs

        action_mask, fling_this = action_mask

        label = label.to(device, non_blocking=True)
        
        obs = obs.to(device, non_blocking=True)
        kp_stack = kp_stack.to(device, non_blocking=True)
        action_prev = action_prev.to(device, non_blocking=True)
        fling_prev = fling_prev.to(device, non_blocking=True)
        action_mask = action_mask.to(device, non_blocking=True)

        # value_pred_dense, fling_pred = value_net(kp_stack, action_prev, fling_prev)
        _, fling_pred = value_net(kp_stack, action_prev, fling_prev)
        # value_pred = torch.masked_select(
        #     value_pred_dense.squeeze(),
        #     action_mask
        #     )
        pred_idx = new_env_utils.fling_params_to_idx(*fling_this.T)
        pred_idx = pred_idx.to(device)
        fling_pred_value = torch.gather(fling_pred, 1, pred_idx[None, :]).flatten()

        # loss1 = criterion(value_pred, label)
        loss2 = criterion(fling_pred_value, label)
        # loss = loss1 + loss2
        loss = loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value_net.steps += 1
        writer.add_scalar(
            f'loss/total/{value_net_key}',
            loss.cpu().item(),
            global_step=value_net.steps)
        writer.add_scalar(
            f'loss/valuemap/{value_net_key}',
            loss1.cpu().item(),
            global_step=value_net.steps)
        writer.add_scalar(
            f'loss/flingmap/{value_net_key}',
            loss2.cpu().item(),
            global_step=value_net.steps)


if __name__ == '__main__':
    args = config_parser().parse_args()
    ray.init(log_to_driver=False)
    seed_all(args.seed)
    policy, optimizer, dataset_path = setup_network(args)
    criterion = torch.nn.functional.mse_loss
    writer = SummaryWriter(logdir=args.log)
    if not os.path.exists(args.log + '/args.pkl'):
        pickle.dump(args, open(args.log + '/args.pkl', 'wb'))
    envs, task_loader = setup_envs(dataset=dataset_path, **vars(args))
    observations = ray.get([e.reset.remote() for e in envs])
    observations = [obs for obs, _ in observations]
    remaining_observations = []
    ready_envs = copy(envs)
    dataset_size = get_dataset_size(dataset_path)
    i = dataset_size
    while(True):
        with torch.no_grad():
            ready_envs, observations, remaining_observations =\
                step_env(
                    all_envs=envs,
                    ready_envs=ready_envs,
                    ready_actions=policy.act(observations),
                    remaining_observations=remaining_observations)
            if i > args.warmup:
                policy.decay_exploration()
        if optimizer is not None and dataset_size > args.warmup:
            if i % args.update_frequency == 0:
                policy.train()
                with FileLock(dataset_path + ".lock"):
                    # for action_primitive, value_net in policy.value_nets.items():
                    optimize(
                        value_net_key='fling',
                        value_net=policy.value_net,
                        optimizer=optimizer,
                        loader=get_loader(
                            hdf5_path=dataset_path,
                            **vars(args)),
                        criterion=criterion,
                        writer=writer,
                        num_updates=args.batches_per_update)
                policy.eval()
            checkpoint_paths = [f'{args.log}/latest_ckpt.pth']
            if i % args.save_ckpt == 0:
                checkpoint_paths.append(
                    f'{args.log}/ckpt_{policy.steps():06d}.pth')
            for path in checkpoint_paths:
                torch.save({'net': policy.state_dict(),
                            'optimizer': optimizer.state_dict()}, path)
        dataset_size = get_dataset_size(dataset_path)
        if i % 32 == 0 and dataset_size > 0:
            stats = collect_stats(dataset_path)
            print('='*18 + f' {dataset_size} points ' + '='*18)
            for key, value in stats.items():
                if '_steps' in key:
                    continue
                elif 'distribution' in key:
                    writer.add_histogram(
                        key, value,
                        global_step=dataset_size)
                elif 'img' in key:
                    writer.add_image(
                        key, value,
                        global_step=dataset_size)
                else:
                    writer.add_scalar(
                        key, float(value),
                        global_step=dataset_size)
                    print(f'\t[{key:<36}]:\t{value:.04f}')
        i += 1
