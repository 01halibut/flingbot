import numpy as np
import torch
from itertools import product
import ray

FLING_HEIGHT_RANGE = (0.04, 0.7)
FLING_SPEED_RANGE = (1e-3, 1e-2)
FLING_LOWER_SPEED_RANGE = (1e-3, 2e-2)
FLING_END_SLACK_RANGE = (0.8, 1)

FLING_HEIGHT_STEPS = 10
FLING_SPEED_STEPS = 10
FLING_LOWER_SPEED_STEPS = 10
FLING_END_SLACK_STEPS = 3

N_KEYPOINTS = 30

def __to_digit(num, min, max, steps):
    out = np.round((num - min) / (max - min) * (steps - 1))
    if type(out) == np.ndarray or type(out) == torch.Tensor:
        return torch.tensor(out, dtype=torch.int64)
    return int(out)

def __from_digit(step, min, max, steps):
    return (max - min) / (steps - 1) * (step) + min

def fling_params_to_idx(fling_height, fling_speed, fling_lower_speed, fling_end_slack):
    height_digit = __to_digit(fling_height, *FLING_HEIGHT_RANGE, FLING_HEIGHT_STEPS)
    speed_digit = __to_digit(fling_speed, *FLING_SPEED_RANGE, FLING_SPEED_STEPS)
    lower_speed_digit = __to_digit(fling_lower_speed, *FLING_LOWER_SPEED_RANGE, FLING_LOWER_SPEED_STEPS)
    end_slack_digit = __to_digit(fling_end_slack, *FLING_END_SLACK_RANGE, FLING_END_SLACK_STEPS)

    out = end_slack_digit
    out = out * FLING_LOWER_SPEED_STEPS + lower_speed_digit
    out = out * FLING_SPEED_STEPS + speed_digit
    out = out * FLING_HEIGHT_STEPS + height_digit
    return out


def idx_to_fling_params(idx):
    height_digit = idx % FLING_HEIGHT_STEPS
    idx = idx // FLING_HEIGHT_STEPS
    speed_digit = idx % FLING_SPEED_STEPS
    idx = idx // FLING_SPEED_STEPS
    lower_speed_digit = idx % FLING_LOWER_SPEED_STEPS
    idx = idx // FLING_LOWER_SPEED_STEPS
    end_slack_digit = idx

    return (__from_digit(height_digit, *FLING_HEIGHT_RANGE, FLING_HEIGHT_STEPS),
            __from_digit(speed_digit, *FLING_SPEED_RANGE, FLING_SPEED_STEPS),
            __from_digit(lower_speed_digit, *FLING_LOWER_SPEED_RANGE, FLING_LOWER_SPEED_STEPS),
            __from_digit(end_slack_digit, *FLING_END_SLACK_RANGE, FLING_END_SLACK_STEPS),
    )


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
    kp_this = np.dot(r_m, kp_this.T).T.astype(np.float32)
    kp_last = np.dot(r_m, kp_last.T).T.astype(np.float32)

    kp_this = torch.tensor(kp_this) * scale
    kp_last = torch.tensor(kp_last) * scale
    kp_stack = torch.cat((kp_this, kp_last), axis=-1)
    return torch.tensor(kp_stack)

make_rotated_scaled_keypoints_async = ray.remote(make_rotated_scaled_keypoints)

def make_rotated_scaled_keypoint_grid(kp_this, kp_last, transformations):
    # kp_stacks = ray.get([
    #     make_rotated_scaled_keypoints_async.remote(kp_this, kp_last, *t) for t in transformations
    # ])
    kp_stacks = [make_rotated_scaled_keypoints(kp_this, kp_last, *t) for t in transformations]
    retval = torch.stack(kp_stacks).float()
    return retval
