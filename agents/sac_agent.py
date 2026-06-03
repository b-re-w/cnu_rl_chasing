from stable_baselines3 import SAC

DEFAULT_SAC_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
)


def make_sac(env, *, seed: int = 0, tensorboard_log: str | None = None, device: str = "auto", **overrides) -> SAC:
    """
    설정된 SAC 모델을 생성

    Args:
        env:             학습 환경(SingleAgentChaseEnv 또는 VecEnv)
        seed:            난수 시드
        tensorboard_log: TensorBoard 로그 디렉터리
        device:          "auto" | "cpu" | "cuda"
        **overrides:     DEFAULT_SAC_KWARGS 덮어쓰기
    """
    kwargs = {**DEFAULT_SAC_KWARGS, **overrides}
    return SAC(env=env, seed=seed, tensorboard_log=tensorboard_log, device=device, verbose=0, **kwargs)
