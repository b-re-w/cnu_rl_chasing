from stable_baselines3 import PPO

DEFAULT_PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def make_ppo(env, *, seed: int = 0, tensorboard_log: str | None = None, device: str = "auto", **overrides) -> PPO:
    """
    설정된 PPO 모델을 생성

    Args:
        env:             학습 환경(SingleAgentChaseEnv 또는 VecEnv)
        seed:            난수 시드
        tensorboard_log: TensorBoard 로그 디렉터리
        device:          "auto" | "cpu" | "cuda"
        **overrides:     DEFAULT_PPO_KWARGS 덮어쓰기
    """
    kwargs = {**DEFAULT_PPO_KWARGS, **overrides}
    return PPO(env=env, seed=seed, tensorboard_log=tensorboard_log, device=device, verbose=0, **kwargs)
