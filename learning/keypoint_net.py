import torch.nn as nn
import torch
from scipy import ndimage as nd
import cv2
from typing import List
import random
from time import time
import ray
import numpy as np

from environment.new_env_utils import *

from learning import nets
fixed_policy = nets.MaximumValuePolicy(
                 action_expl_prob = 0,
                 action_expl_decay = 0,
                 value_expl_prob = 0,
                 value_expl_decay  = 0,
                 action_primitives = ['fling'],
                 num_rotations=12,
                 scale_factors=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75],
                 obs_dim=64,
                 pix_grasp_dist=8,
                 pix_place_dist=10,
                 pix_drag_dist=10,
                 rgb_only=True,
                 depth_only=False,
                 device=torch.zeros(0).cuda().device)
fixed_net = fixed_policy.value_nets['fling']
ckpt = torch.load('flingbot.pth')
fixed_policy.load_state_dict(ckpt['net'])
fixed_value_net = fixed_policy.value_nets['fling']
for param in fixed_policy.parameters():
    param.requires_grad = False
fixed_net.eval().cuda()

def run_fixed_net_inference(obs):
    return fixed_net(obs)

class Policy:
    def __init__(self,
                 action_primitives: List[str],
                 num_rotations: int,
                 scale_factors: List[float],
                 obs_dim: int,
                 pix_grasp_dist: int,
                 pix_drag_dist: int,
                 pix_place_dist: int,
                 **kwargs):
        assert len(action_primitives) > 0
        self.action_primitives = action_primitives
        print('[Policy] Action primitives:')
        for ap in self.action_primitives:
            print(f'\t{ap}')

        # rotation angle in degrees, counter-clockwise
        self.rotations = [(2*i/(num_rotations-1) - 1) * 90
                          for i in range(num_rotations)]
        self.scale_factors = scale_factors
        self.num_transforms = len(self.rotations) * len(self.scale_factors)
        self.obs_dim = obs_dim
        self.pix_grasp_dist = pix_grasp_dist
        self.pix_drag_dist = pix_drag_dist
        self.pix_place_dist = pix_place_dist

    def get_action_single(self, obs):
        raise NotImplementedError()

    def act(self, obs):
        return [self.get_action_single(o) for o in obs]

class MaximumValuePolicyParameterizedFling(nn.Module, Policy):
    def __init__(self,
                 action_expl_prob: float,
                 action_expl_decay: float,
                 value_expl_prob: float,
                 value_expl_decay: float,
                 value_flingbot_weight : float,
                 value_flingbot_decay : float,
                 device=None,
                 **kwargs):
        super().__init__()
        Policy.__init__(self, **kwargs)
        if device is None:
            self.device = torch.device('cuda') \
                if torch.cuda.is_available()\
                else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.action_expl_prob = nn.parameter.Parameter(
            torch.tensor(action_expl_prob), requires_grad=False)
        self.action_expl_decay = nn.parameter.Parameter(
            torch.tensor(action_expl_decay), requires_grad=False)
        self.value_expl_prob = nn.parameter.Parameter(
            torch.tensor(value_expl_prob), requires_grad=False)
        self.value_expl_decay = nn.parameter.Parameter(
            torch.tensor(value_expl_decay), requires_grad=False)
        self.value_flingbot_weight = nn.parameter.Parameter(
            torch.tensor(value_flingbot_weight), requires_grad=False)
        self.value_flingbot_decay = nn.parameter.Parameter(
            torch.tensor(value_flingbot_decay), requires_grad=False)

        # base value net loaded from flingbot base
        # self.base_value_net = SpatialValueNet(device=self.device, **kwargs).to(self.device)
        self.value_net = KpValueNet(N_KEYPOINTS, device=self.device)

        self.should_explore_action = lambda: \
            self.action_expl_prob > random.random()
        self.should_explore_value = lambda: \
            self.value_expl_prob > random.random()

        self.eval()

    def decay_exploration(self):
        self.action_expl_prob *= self.action_expl_decay
        self.value_expl_prob *= self.value_expl_decay
        self.value_flingbot_weight *= self.value_flingbot_decay

    def random_value_map(self):
        return torch.rand(len(self.rotations) * len(self.scale_factors),
                          self.obs_dim, self.obs_dim)

    def get_action_single(self, obs):
        with torch.no_grad():
            transformed_obs, kp_stack, last_action, last_fling = obs
            n_transforms = transformed_obs.shape[0]
            if kp_stack is not None:
                last_fling =  last_fling.float()
                assert transformed_obs.shape[0] == kp_stack.shape[0]               
                kp_stack = kp_stack.to(self.device, non_blocking=True)
                last_action = torch.stack([last_action] * n_transforms)
                last_action = last_action.to(self.device, non_blocking=True)
                last_fling = torch.stack([last_fling] * n_transforms)
                last_fling = last_fling.to(self.device, non_blocking=True)

            transformed_obs = transformed_obs.to(self.device, non_blocking=True)

            if last_action is None:
                assert kp_stack is None
                assert last_fling is None
                value_maps = {'fling': run_fixed_net_inference(transformed_obs).cpu().squeeze()
                            if not self.should_explore_value()
                            else self.random_value_map()}
                fling_heights = 0.3 # 0.04 to 0.7
                fling_speeds = 6e-3 # 1e-3 to 1e-2
                fling_lower_speeds = 1e-2 # 1e-3 to 2e-2
                fling_end_slacks = 1 # 0.8 to 1
                fling_params = np.array([fling_heights, fling_speeds, fling_lower_speeds, fling_end_slacks])
            else:
                assert kp_stack is not None
                assert last_fling is not None
                if self.should_explore_value():
                    fling_heights = np.random.uniform(*FLING_HEIGHT_RANGE)
                    fling_speeds = np.random.uniform(*FLING_SPEED_RANGE)
                    fling_lower_speeds = np.random.uniform(*FLING_LOWER_SPEED_RANGE)
                    fling_end_slacks = np.random.uniform(*FLING_END_SLACK_RANGE)
                    fling_params = np.array([fling_heights, fling_speeds, fling_lower_speeds, fling_end_slacks])
                    value_maps = {'fling': self.random_value_map()}
                else:
                    value_maps, fling_params = self.value_net(kp_stack, last_action, last_fling)
                    value_maps = value_maps.cpu().squeeze()
                    fling_params = fling_params.cpu().squeeze().numpy()
                    value_maps_2 = run_fixed_net_inference(transformed_obs).cpu().squeeze()
                    value_maps = (value_maps * (0.5 + 0.5 * (self.value_flingbot_weight)) +
                                  value_maps_2 * (0.5 + 0.5 * (1 - self.value_flingbot_weight))
                                 )
                    value_maps = {'fling' : value_maps}

            if self.should_explore_action():
                random_action, action_val_map = random.choice(
                    list(value_maps.items()))
                min_val = action_val_map.min()
                value_maps = {
                    key: (val_map
                          if key == random_action
                          else torch.ones(val_map.size()) * min_val)
                    for key, val_map in value_maps.items()}
            return (value_maps,
                fling_params
            )

    def steps(self):
        return self.value_net.steps.cpu().detach().item()

    def forward(self, obs):
        return self.act(obs)


class SelfAttention(nn.Module):
    def __init__(self, input_channels, num_heads):
        super().__init__()
        self.key_gate = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=1),
            torch.nn.BatchNorm1d(input_channels),
            torch.nn.ReLU()
        )

        self.value_gate = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=1),
            torch.nn.BatchNorm1d(input_channels),
            torch.nn.ReLU()
        )

        self.query_gate = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=1),
            torch.nn.BatchNorm1d(input_channels),
            torch.nn.ReLU()
        )

        self.attention = torch.nn.MultiheadAttention(
            input_channels, num_heads, batch_first = True
        )

    def forward(self, x):
        keys = self.key_gate(x)
        queries = self.query_gate(x)
        values = self.value_gate(x)
        keys = torch.swapaxes(keys, -1, -2)
        queries = torch.swapaxes(queries, -1, -2)
        values = torch.swapaxes(values, -1, -2)
        attended = self.attention(queries, keys, values, need_weights=False)
        return torch.swapaxes(attended[0], -1, -2)


class KpValueNet(nn.Module):
    def __init__(self, n_keypoints, device='cuda', steps=0):
        super().__init__()

        self.n_keypoints = n_keypoints

        self.device = device

        self.kp_in_gate = torch.nn.Sequential(
            # torch.nn.Flatten(start_dim=0, end_dim=1), # (batch x rotations) x 4 x keypoints jk it's actually just batch x rotations x 4
            torch.nn.Conv1d(in_channels=4, out_channels=32, kernel_size=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
        )

        self.kp_attention = SelfAttention(32, 4)

        self.attn_gate_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
        )

        self.kp_attention_2 = SelfAttention(32, 4)

        self.attn_gate_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
        )
        
        # Swap channels and keypoints before this, input (batch x rotations) x keypoints x 32
        self.kp_out_gate = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(start_dim=1, end_dim=2),
            torch.nn.Linear(self.n_keypoints, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU()
        )

        self.action_in_gate = torch.nn.Sequential(
            torch.nn.Flatten()
        )

        self.fing_params_in_gate = torch.nn.Sequential(
            torch.nn.Linear(4, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU()
        )

        self.out_gate1 = torch.nn.Sequential(
            torch.nn.Linear(8192, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.Unflatten(-1, (64, 64))
        )

        self.out_gate2 = torch.nn.Sequential(
            torch.nn.Linear(8192, 3000),
            torch.nn.BatchNorm1d(3000),
            torch.nn.ReLU(),
            torch.nn.Linear(3000, 3000)
        )

        
        self.steps = nn.parameter.Parameter(
            torch.tensor(steps), requires_grad=False)

    def forward(self,
                kp_stack,
                action_prev,
                last_fling):

        kp_stack = torch.swapaxes(kp_stack, -1, -2) # batch x 4 x keypoints

        kp_e = self.kp_in_gate(kp_stack)
        kp_e = self.kp_attention(kp_e)
        kp_e = self.attn_gate_1(kp_e)
        kp_e = self.kp_attention_2(kp_e)
        kp_e = self.attn_gate_2(kp_e)
        kp_e = torch.swapaxes(kp_e, -1, -2)
        kp_e = self.kp_out_gate(kp_e)

        action_e = self.action_in_gate(action_prev)

        fling_e = self.fing_params_in_gate(last_fling)

        full_embed = torch.cat((kp_e, action_e, fling_e), dim=1)

        out1 = self.out_gate1(full_embed)
        out2 = self.out_gate2(full_embed)

        return out1, out2

        