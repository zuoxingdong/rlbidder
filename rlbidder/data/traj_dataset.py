import logging
import warnings
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def right_zero_pad(arr: np.ndarray, pad_to: int) -> np.ndarray:
    pad_width = pad_to - arr.shape[0]
    if pad_width <= 0:
        return arr
    return np.pad(arr, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)


class TrajDataset(Dataset):
    def __init__(
        self,
        traj_path: str,
        seq_len: int = 10,
        normalize_states: bool = False,
        seed: int = 42,
        sample_offset: int = 5,
    ) -> None:
        self.seq_len = seq_len
        self.normalize_states = normalize_states
        self.sample_offset = sample_offset
        if normalize_states:
            msg = "Offline dataset states already processed via scalers. Further normalization not recommended."
            logger.warning(msg)
            warnings.warn(msg, UserWarning)
        
        self.dataset = np.load(traj_path, allow_pickle=True)
        self.states = self.dataset["states"]
        self.actions = self.dataset["actions"]
        self.rewards = self.dataset["rewards"]
        self.dones = self.dataset["dones"]
        self.rtgs = self.dataset["rtgs"]
        self.penalties = self.dataset["penalties"]
        
        self.traj_lens = self.dataset["traj_lens"].astype(np.float32)
        self.state_mean = self.dataset["state_mean"].astype(np.float32)
        self.state_std = self.dataset["state_std"].astype(np.float32)
        
        self.rng = np.random.Generator(np.random.PCG64(seed))
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = self.traj_lens / self.traj_lens.sum()

    def __prepare_sample(self, traj_idx: int, start_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Each field is an array of object dtype, where each element is a trajectory (array)
        states = self.states[traj_idx][start_idx : start_idx + self.seq_len]
        actions = self.actions[traj_idx][start_idx : start_idx + self.seq_len]
        rewards = self.rewards[traj_idx][start_idx : start_idx + self.seq_len]
        penalties = self.penalties[traj_idx][start_idx : start_idx + self.seq_len]
        dones = self.dones[traj_idx][start_idx : start_idx + self.seq_len]
        rtgs = self.rtgs[traj_idx][start_idx : start_idx + self.seq_len + 1]  # +1 for target prediction
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        # TODO: should be removed, scaler and data processing handled separately
        if self.normalize_states:
            raise ValueError("Normalization should be handled separately.")
            # states = (states - self.state_mean) / self.state_std
        
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[: len(states)] = 1.0
        
        states = right_zero_pad(states, pad_to=self.seq_len)
        actions = right_zero_pad(actions[:, None], pad_to=self.seq_len)
        rewards = right_zero_pad(rewards[:, None], pad_to=self.seq_len).squeeze(1)
        penalties = right_zero_pad(penalties[:, None], pad_to=self.seq_len).squeeze(1)
        dones = right_zero_pad(dones[:, None], pad_to=self.seq_len).squeeze(1)
        rtgs = right_zero_pad(rtgs[:, None], pad_to=self.seq_len + 1).squeeze(1)  # pad to seq_len + 1

        return (
            states.astype(np.float32), 
            actions.astype(np.float32), 
            rewards.astype(np.float32), 
            penalties.astype(np.float32),
            dones.astype(np.float32),
            rtgs.astype(np.float32), 
            time_steps.astype(int), 
            mask.astype(bool)
        )

    def __len__(self) -> int:
        return int(self.traj_lens.sum())

    # TODO: vectorize __getitem__ to speed up sampling of starting index 
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        traj_idx = index
        max_start_idx = max(1, self.traj_lens[traj_idx] - self.sample_offset)
        start_idx = self.rng.integers(0, max_start_idx)
        return self.__prepare_sample(traj_idx, start_idx)
