import numpy as np
import torch
from itertools import product
import ray

FLING_HEIGHT_RANGE = (0.04, 0.7)
FLING_SPEED_RANGE = (1e-3, 1e-2)
FLING_LOWER_SPEED_RANGE = (1e-3, 2e-2)
FLING_END_SLACK_RANGE = (0.8, 1)

N_KEYPOINTS = 30

def pick_random_keypoints(last_state, this_state, n_keypoints=N_KEYPOINTS):
    assert last_state.shape[0] == this_state.shape[0]
    idxr = np.random.randint(0, last_state.shape[0], n_keypoints)
    last_state = torch.tensor(np.array(last_state)[idxr][:,::2])
    this_state = torch.tensor(np.array(this_state)[idxr][:,::2])
    return last_state, this_state

def make_rotated_scaled_keypoints(kp_this, kp_last, rotation, scale):
    cos = np.cos(rotation)
    sin = np.sin(rotation)
    r_m = np.array([
        [cos, -sin],
        [sin, cos]
    ])
    kp_this = torch.tensor(np.dot(r_m, kp_this.T).T) * scale
    kp_last = torch.tensor(np.dot(r_m, kp_last.T).T) * scale
    kp_stack = torch.cat((kp_this, kp_last), axis=-1)
    print(kp_this.shape, kp_last.shape, kp_stack.shape)
    return torch.tensor(kp_stack)

make_rotated_scaled_keypoints_async = ray.remote(make_rotated_scaled_keypoints)

def make_rotated_scaled_keypoint_grid(kp_this, kp_last, transformations):
    # kp_stacks = ray.get([
    #     make_rotated_scaled_keypoints_async.remote(kp_this, kp_last, *t) for t in transformations
    # ])
    kp_stacks = [make_rotated_scaled_keypoints(kp_this, kp_last, *t) for t in transformations]
    retval = torch.stack(kp_stacks).float()
    return retval
