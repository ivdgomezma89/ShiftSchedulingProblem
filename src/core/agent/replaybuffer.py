
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, Optional


class ReplayBuffer:
    """Buffer de experiencia para DQN"""

    def __init__(self, buffer_size: int, observation_space: spaces.Space,
                 action_space: spaces.Space, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Determinar si usamos máscaras
        self.use_masking = isinstance(observation_space, spaces.Dict) and 'action_mask' in observation_space.spaces

        if self.use_masking:
            obs_shape = observation_space['observation'].shape
            mask_shape = observation_space['action_mask'].shape
        else:
            obs_shape = observation_space.shape
            mask_shape = None

        # Inicializar buffers
        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        if self.use_masking:
            self.action_masks = np.zeros((buffer_size,) + mask_shape, dtype=np.float32)
            self.next_action_masks = np.zeros((buffer_size,) + mask_shape, dtype=np.float32)
        else:
            self.action_masks = None
            self.next_action_masks = None

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool,
            action_mask: Optional[np.ndarray] = None, next_action_mask: Optional[np.ndarray] = None):
        """Añadir una transición al buffer"""

        if self.use_masking:
            if isinstance(obs, dict):
                self.observations[self.ptr] = obs['observation']
                self.action_masks[self.ptr] = obs['action_mask']
            else:
                raise ValueError("Se esperaba observación tipo dict con action_mask")

            if isinstance(next_obs, dict):
                self.next_observations[self.ptr] = next_obs['observation']
                self.next_action_masks[self.ptr] = next_obs['action_mask']
            else:
                raise ValueError("Se esperaba next_obs tipo dict con action_mask")
        else:
            self.observations[self.ptr] = obs
            self.next_observations[self.ptr] = next_obs

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Muestrear un batch de transiciones"""
        indices = np.random.choice(self.size, batch_size, replace=False)

        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
        }

        if self.use_masking:
            batch['action_masks'] = torch.FloatTensor(self.action_masks[indices]).to(self.device)
            batch['next_action_masks'] = torch.FloatTensor(self.next_action_masks[indices]).to(self.device)

        return batch

