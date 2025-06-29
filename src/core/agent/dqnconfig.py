
from dataclasses import dataclass
from typing import  Optional

@dataclass
class DQNConfig:
    """Configuración para el algoritmo DQN Maskeable"""
    learning_rate: float = 1e-4
    buffer_size: int = 100000
    learning_starts: int = 500
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0
    seed: Optional[int] = None


