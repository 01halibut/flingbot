import torch.nn as nn
import torch
from scipy import ndimage as nd
import cv2
from typing import List
import random
from time import time
import ray
import numpy as np

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
        if 'fling' not in action_primitives:
            self.rotations = [(2*i/num_rotations - 1) *
                              180 for i in range(num_rotations)]
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


class MaximumValuePolicy(nn.Module, Policy):
    def __init__(self,
                 action_expl_prob: float,
                 action_expl_decay: float,
                 value_expl_prob: float,
                 value_expl_decay: float,
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

        # one value net per action primitive
        self.value_net = SpatialValueNet(device=self.device, **kwargs).to(self.device)
        self.should_explore_action = lambda: \
            self.action_expl_prob > random.random()
        self.should_explore_value = lambda: \
            self.value_expl_prob > random.random()

        self.eval()

    def decay_exploration(self):
        self.action_expl_prob *= self.action_expl_decay
        self.value_expl_prob *= self.value_expl_decay

    def random_value_map(self):
        return torch.rand(len(self.rotations) * len(self.scale_factors),
                          self.obs_dim, self.obs_dim)

    def get_action_single(self, obs):
        with torch.no_grad():
            obs = obs.to(self.device, non_blocking=True)
            value_maps = {key: val_net(obs).cpu().squeeze()
                          if not self.should_explore_value()
                          else self.random_value_map()

                          for key, val_net in self.value_nets.items()}
            if self.should_explore_action():
                random_action, action_val_map = random.choice(
                    list(value_maps.items()))
                min_val = action_val_map.min()
                value_maps = {
                    key: (val_map
                          if key == random_action
                          else torch.ones(val_map.size()) * min_val)
                    for key, val_map in value_maps.items()}
            return value_maps

    def steps(self):
        return sum([net.steps for net in self.value_nets.values()])

    def forward(self, obs):
        return self.act(obs)
