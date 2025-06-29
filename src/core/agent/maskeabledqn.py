
from .replaybuffer import ReplayBuffer
from .dqnconfig import DQNConfig
from .qnetwork import QNetwork


import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque



class MaskeableDQN:
    """Implementación de DQN con soporte para máscaras de acción"""

    def __init__(self,
                 env: gym.Env,
                 config: DQNConfig = None,
                 policy_kwargs: Dict[str, Any] = None,
                 device: str = "auto",
                 verbose: int = 0):

        self.env = env
        self.config = config if config is not None else DQNConfig()
        self.verbose = verbose

        # Configurar device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.config.seed is not None:
            self.set_random_seed(self.config.seed)

        # Configurar espacios
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Determinar si usamos máscaras
        self.use_masking = isinstance(self.observation_space, spaces.Dict) and 'action_mask' in self.observation_space.spaces

        # Configurar redes
        policy_kwargs = policy_kwargs or {}
        net_arch = policy_kwargs.get('net_arch', [64, 64])


        self.q_net = QNetwork(self.observation_space, self.action_space, net_arch).to(self.device)
        self.q_net_target = QNetwork(self.observation_space, self.action_space, net_arch).to(self.device)

        # Sincronizar redes
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Optimizador
        self.optimizer = Adam(self.q_net.parameters(), lr=self.config.learning_rate)

        # Buffer de experiencia
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size,
            self.observation_space,
            self.action_space,
            self.device
        )

        # Variables de entrenamiento
        self.num_timesteps = 0
        self.learning_starts = self.config.learning_starts
        self._episode_rewards = deque(maxlen=300000)
        self._episode_lengths = deque(maxlen=300000)

        # Epsilon para exploración
        self.exploration_schedule = self._get_linear_schedule(
            self.config.exploration_initial_eps,
            self.config.exploration_final_eps,
            self.config.exploration_fraction
        )

    def set_random_seed(self, seed: int):
        """Establecer semillas para reproducibilidad"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _get_linear_schedule(self, initial_value: float, final_value: float, fraction: float):
        """Crear un schedule lineal para epsilon"""
        def schedule(progress: float) -> float:
            if progress > fraction:
                return final_value
            else:
                return initial_value + (final_value - initial_value) * (progress / fraction)
        return schedule

    def predict(self, observation: np.ndarray, deterministic: bool = False,
                action_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predecir acción dado una observación"""

        # **MODIFICATION START**
        # Handle dictionary observation format provided by the environment
        if self.use_masking and isinstance(observation, dict):
            obs_tensor = torch.FloatTensor(observation['observation']).unsqueeze(0).to(self.device)
            # Use the mask from the observation dictionary if action_mask param is None
            if action_mask is None:
                mask_tensor = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)
            else:
                 # If an explicit mask is provided (e.g., for evaluation), use it
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        # Handle standard non-dictionary observation (for non-masking envs or specific use cases)
        else:
            # The error was here: torch.FloatTensor expects a sequence, not a dict.
            # This case should ideally only be hit if observation is already a sequence.
            # Ensure observation is treated as a sequence here.
            # It's safer to assume it's a numpy array or tensor if not a dict.
            if isinstance(observation, np.ndarray):
                 obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            elif isinstance(observation, torch.Tensor):
                 obs_tensor = observation.unsqueeze(0).to(self.device) # Add batch dim if needed
            else:
                 # Fallback or error if observation type is unexpected
                 raise TypeError(f"Unsupported observation type: {type(observation)}")

            mask_tensor = None # No mask is available in this case

        # **MODIFICATION END**

        # if self.use_masking and isinstance(observation, dict):
        #     obs_tensor = torch.FloatTensor(observation['observation']).unsqueeze(0).to(self.device)
        #     mask_tensor = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)
        # else:
        #     obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        #     mask_tensor = None

        with torch.no_grad():
            q_values = self.q_net(obs_tensor, mask_tensor)

        if deterministic:
            action = q_values.argmax(dim=1).cpu().numpy()
        else:
            # Epsilon-greedy con máscara
            epsilon = self.exploration_schedule(self.num_timesteps / self.learning_starts)

            if np.random.random() < epsilon:
                if mask_tensor is not None:
                    # Muestrear solo de acciones válidas
                    valid_actions = torch.where(mask_tensor[0] == 1)[0]
                    if len(valid_actions) > 0:
                        action_idx = np.random.choice(len(valid_actions))
                        action = valid_actions[action_idx].cpu().numpy()
                    else:
                        action = np.array([0])  # Fallback
                else:
                    action = np.array([np.random.randint(self.action_space.n)])
            else:
                action = q_values.argmax(dim=1).cpu().numpy()

        return action, None
    
    def evaluate_acceptance_criterio(self, best_tuple, test_tuple):
        " función para evaluar si se acepta una solución con base"
        " en los criterios de prioridad (lexicograficamente)"
        return test_tuple > best_tuple

    def learn(self, total_timesteps: int, progress_dict_drl,log_interval: int = 1000, **kwargs) -> 'MaskeableDQN':
        """Entrenar el modelo"""

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        self.best_solutions={'best_emp_no_asignados':{'solution':None, 'reward':(-np.inf, -np.inf)},
                      'best_dias_pref':{'solution':None, 'reward':(-np.inf, -np.inf)},
                      'best_desk_consistency':{'solution':None, 'reward':(-np.inf, -np.inf)},
                      'best_team_proximity':{'solution':None, 'reward':(-np.inf, -np.inf)},
                      'best_asistencia_equipos':{'solution':None, 'reward':(-np.inf, -np.inf)}
                      }
        for step in range(total_timesteps):
            progress_dict_drl['drl'] = step+1
            self.num_timesteps += 1

            # Seleccionar acción
            action_array, _ = self.predict(obs, deterministic=False)
            action = action_array.item()

            # Ejecutar acción
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Almacenar transición
            self.replay_buffer.add(obs, action, reward, next_obs, done)

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            # Reset si el episodio terminó
            if done:
                self._episode_rewards.append(episode_reward)
                self._episode_lengths.append(episode_length)

                sol_test_empl= (self.env.emp_no_asignados, self.env.dias_pref + self.env.desk_consistency + self.env.team_proximity + self.env.asistencia_equipos)
                sol_test_dias_pref= (self.env.dias_pref, self.env.emp_no_asignados + self.env.desk_consistency + self.env.team_proximity + self.env.asistencia_equipos)
                sol_test_desk_consistency= (self.env.desk_consistency, self.env.emp_no_asignados + self.env.dias_pref + self.env.team_proximity + self.env.asistencia_equipos)
                sol_test_team_proximity= (self.env.team_proximity, self.env.emp_no_asignados + self.env.dias_pref + self.env.desk_consistency + self.env.asistencia_equipos)
                sol_test_asistencia_equipos= (self.env.asistencia_equipos, self.env.emp_no_asignados + self.env.dias_pref + self.env.desk_consistency + self.env.team_proximity)

                if self.evaluate_acceptance_criterio(self.best_solutions['best_emp_no_asignados']['reward'], sol_test_empl):          
                    self.best_solutions['best_emp_no_asignados']['reward']= sol_test_empl
                    self.best_solutions['best_emp_no_asignados']['solution']= self.env._state_matrix.copy()
                if self.evaluate_acceptance_criterio(self.best_solutions['best_dias_pref']['reward'], sol_test_dias_pref):
                    self.best_solutions['best_dias_pref']['reward']= sol_test_dias_pref
                    self.best_solutions['best_dias_pref']['solution']= self.env._state_matrix.copy()
                if self.evaluate_acceptance_criterio(self.best_solutions['best_desk_consistency']['reward'], sol_test_desk_consistency):
                    self.best_solutions['best_desk_consistency']['reward']= sol_test_desk_consistency
                    self.best_solutions['best_desk_consistency']['solution']= self.env._state_matrix.copy()
                if self.evaluate_acceptance_criterio(self.best_solutions['best_team_proximity']['reward'], sol_test_team_proximity):
                    self.best_solutions['best_team_proximity']['reward']= sol_test_team_proximity
                    self.best_solutions['best_team_proximity']['solution']= self.env._state_matrix.copy()
                if self.evaluate_acceptance_criterio(self.best_solutions['best_asistencia_equipos']['reward'], sol_test_asistencia_equipos):
                    self.best_solutions['best_asistencia_equipos']['reward']= sol_test_asistencia_equipos
                    self.best_solutions['best_asistencia_equipos']['solution']= self.env._state_matrix.copy()


                # if self.env.dias_pref > self.best_solutions['best_dias_pref']['reward']:
                #     self.best_solutions['best_dias_pref']['reward']= self.env.dias_pref
                #     self.best_solutions['best_dias_pref']['solution']= self.env._state_matrix.copy()
                # if self.env.desk_consistency > self.best_solutions['best_desk_consistency']['reward']:
                #     self.best_solutions['best_desk_consistency']['reward']= self.env.desk_consistency
                #     self.best_solutions['best_desk_consistency']['solution']= self.env._state_matrix.copy()
                # if self.env.team_proximity > self.best_solutions['best_team_proximity']['reward']:
                #     self.best_solutions['best_team_proximity']['reward']= self.env.team_proximity
                #     self.best_solutions['best_team_proximity']['solution']= self.env._state_matrix
                # if self.env.asistencia_equipos > self.best_solutions['best_asistencia_equipos']['reward']:
                #     self.best_solutions['best_asistencia_equipos']['reward']= self.env.asistencia_equipos
                #     self.best_solutions['best_asistencia_equipos']['solution']= self.env._state_matrix.copy()

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

            # Entrenar si tenemos suficientes muestras
            if (self.num_timesteps >= self.learning_starts and
                self.num_timesteps % self.config.train_freq == 0):
                self._train()

            # Actualizar red target
            if self.num_timesteps % self.config.target_update_interval == 0:
                self._update_target_network()

            # Log
            if (self.verbose > 0 and step % log_interval == 0 and
                len(self._episode_rewards) > 0):
                mean_reward = np.mean(self._episode_rewards)
                print(f"Step: {step}, Mean Reward: {mean_reward:.2f}")


        return self

    def _train(self):
        """Realizar un paso de entrenamiento"""
        if self.replay_buffer.size < self.config.batch_size:
            return

        # Muestrear batch
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Calcular Q-values actuales
        if self.use_masking:
            current_q_values = self.q_net(batch['observations'], batch['action_masks'])
        else:
            current_q_values = self.q_net(batch['observations'])

        current_q_values = current_q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze(1)

        # Calcular Q-values target
        with torch.no_grad():
            if self.use_masking:
                next_q_values = self.q_net_target(batch['next_observations'], batch['next_action_masks'])
            else:
                next_q_values = self.q_net_target(batch['next_observations'])

            next_q_values = next_q_values.max(1)[0]
            target_q_values = batch['rewards'] + (1 - batch['dones']) * self.config.gamma * next_q_values

        # Calcular loss
        loss = F.mse_loss(current_q_values, target_q_values)
        # Huber loss para mayor estabilidad
        #loss = F.smooth_l1_loss(current_q_values, target_q_values)


        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

    def _update_target_network(self):
        """Actualizar red target"""
        if self.config.tau == 1.0:
            # Hard update
            self.q_net_target.load_state_dict(self.q_net.state_dict())
        else:
            # Soft update
            for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

    def save(self, path: str):
        """Guardar modelo"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'q_net_target_state_dict': self.q_net_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'num_timesteps': self.num_timesteps
        }, path)

    def load(self, path: str):
        """Cargar modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint['num_timesteps']




