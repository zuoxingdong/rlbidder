import numpy as np
from torch.utils.data import Dataset


class OfflineReplayBuffer(Dataset):
    """
    Circular replay buffer for off-policy RL algorithms, compatible with torch Dataset.
    """
    def __init__(self, buffer_size: int, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

        self.states = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((buffer_size), dtype=np.float32)
        self.cpa_compliance_ratios = np.zeros((buffer_size), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        cpa_compliance_ratio: np.ndarray,
    ) -> None:
        """
        Add a batch of transitions to the buffer.
        All arguments should be arrays of shape (batch_size, ...).
        """
        batch_size = state.shape[0]
        idxs = np.arange(self.pos, self.pos + batch_size) % self.buffer_size

        self.states[idxs] = state
        self.actions[idxs] = action
        self.rewards[idxs] = reward
        self.next_states[idxs] = next_state
        self.dones[idxs] = done
        self.cpa_compliance_ratios[idxs] = cpa_compliance_ratio

        self.pos = (self.pos + batch_size) % self.buffer_size
        if not self.full and self.pos == 0:
            self.full = True

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.states[idx], 
            self.actions[idx], 
            self.rewards[idx],
            self.next_states[idx], 
            self.dones[idx], 
            self.cpa_compliance_ratios[idx],
        )


