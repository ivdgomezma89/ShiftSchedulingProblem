import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from gymnasium import spaces


class QNetwork(nn.Module):
    """Red neuronal Q con soporte para máscaras de acción"""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Discrete,
                 net_arch: List[int] = [64, 64]):
        super().__init__()

        # Determinar el tamaño de entrada
        if isinstance(observation_space, spaces.Box):
            obs_dim = observation_space.shape[0]
        elif isinstance(observation_space, spaces.Dict):
            # Asumir que hay 'observation' y opcionalmente 'action_mask'
            obs_dim = observation_space['observation'].shape[0]
        else:
            raise ValueError(f"Tipo de observation_space no soportado: {type(observation_space)}")

        self.action_dim = action_space.n

        # Crear la red
        layers = []
        input_dim = obs_dim

        for hidden_dim in net_arch:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.action_dim))

        self.q_net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass con aplicación opcional de máscara de acción"""
        q_values = self.q_net(obs)

        if action_mask is not None:
            # Aplicar máscara: poner -inf a acciones inválidas
            masked_q_values = q_values.clone()
            masked_q_values[action_mask == 0] = -np.inf
            return masked_q_values

        return q_values