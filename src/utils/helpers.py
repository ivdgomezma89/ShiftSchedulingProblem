def optimize_dqn_hyperparams(env, trial: optuna.Trial, env_fn, n_timesteps: int = 50000,
                        n_eval_episodes: int = 1) -> float:
    """Función objetivo para optimización de hiperparámetros con Optuna"""

    # Sugerir hiperparámetros
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 100, 30000, step=50)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.05, 0.3)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
    target_update_interval = trial.suggest_categorical('target_update_interval', [500, 1000, 3000, 5000, 8000, 10000, 15000,20000])
    net_arch_size = trial.suggest_int('net_arch_size', 20, 100, step=5)
    net_arch_layers = trial.suggest_int('net_arch_layers', 2, 8, step=1)

    # Crear configuración
    config = DQNConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        learning_starts=min(1000, n_timesteps // 10),
        seed=42,
    )

    policy_kwargs = {
        'net_arch': [net_arch_size] * net_arch_layers
    }

    # Entrenar modelo
    model = MaskableDQN(env, config=config, policy_kwargs=policy_kwargs, verbose=0)
    model.learn(total_timesteps=n_timesteps)

    # Evaluar modelo
    total_reward = 0
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward

    mean_reward = total_reward / n_eval_episodes
    env.close()

    return mean_reward


def run_optuna_optimization(env, n_trials: int = 100, n_timesteps: int = 50000) -> optuna.Study:
    """Ejecutar optimización de hiperparámetros con Optuna"""

    study = optuna.create_study(direction='maximize')

    def objective(trial):
        return optimize_dqn_hyperparams(env, trial, n_timesteps)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True )

    print("Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"Mejor puntuación: {study.best_value}")

    return study
