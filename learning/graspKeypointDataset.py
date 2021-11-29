import torch
from torchvision import transforms
import h5py
from tqdm import tqdm
import numpy as np

REWARDS_MEAN = 0.0029411377084902638
REWARDS_STD = 0.011524952525922203
REWARDS_MAX = 0.20572495126190674
REWARDS_MIN = -0.11034914070874759


class GraspKeypointDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hdf5_path: str,
                 use_normalized_coverage=True,
                 num_keypoints = 20,
                 **kwargs):
        self.hdf5_path = hdf5_path
        self.use_normalized_coverage = use_normalized_coverage

        self.keys = self.get_key_pairs()
        self.size = len(self.keys)

        self.num_keypoints = num_keypoints
    
    def __make_key_pair(self, key):
        tokens = key.split('_')
        num = int(tokens[1][4:])
        if num == 0:
            return None
        return f"{tokens[0]}_step{num - 1:02d}"

    def get_key_pairs(self):
        with h5py.File(self.hdf5_path, "r") as dataset:
            keys = []
            for k in dataset:
                group = dataset[k]
                prev = self.__make_key_pair(k)
                if (prev != None
                    and 'fling_height' in group.attrs.keys()
                    and 'fling_speed' in group.attrs.keys()
                    and 'fling_lower_speed' in group.attrs.keys()
                    and 'fling_end_slack' in group.attrs.keys()
                ):
                    keys.append((k, prev))
            return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as dataset:
            key, key_prev = self.keys[index]
            group = dataset.get(key)
            group_prev = dataset.get(key_prev)
            reward = float(group.attrs['postaction_coverage']
                           - group.attrs['preaction_coverage'])
            if self.use_normalized_coverage:
                reward /= float(group.attrs['max_coverage'])
            else:
                reward = (reward - REWARDS_MIN) /\
                    (REWARDS_MAX - REWARDS_MIN)
            
            state = group['state']
            state_prev = group_prev['state']

            idxr = np.random.randint(0, state.shape[0], self.num_keypoints)
            kp      = torch.tensor(np.array(state)[idxr])
            kp_prev = torch.tensor(np.array(state_prev)[idxr])

            action_prev = torch.tensor(group_prev['actions']).bool()

            obs = (torch.tensor(group['observations']),
                    kp,
                    kp_prev,
                    action_prev
                  )

            action = (
                torch.tensor(group['actions']).bool(),
                group.attrs['fling_height'],
                group.attrs['fling_speed'],
                group.attrs['fling_lower_speed'],
                group.attrs['fling_end_slack'],
            )

            reward = torch.tensor(reward).float()

            return (obs, action, reward)
