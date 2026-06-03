"""
Residual 보행: 스크립트(open-loop) 보행 위에 RL 이 '조향 잔차'만 얹는 방식

배경:
    Ant 는 (1) 걷기와 (2) 추격/회피를 동시에 백지에서 배워야 해서 학습이 느리고
    "제자리 정지" 지역최적해에 잘 빠진다. 이 모듈은 검증된 직진 보행기(CPG)를
    '운동 prior' 로 깔아두고, RL 은 그 위에 더해질 작은 보정(잔차)만 학습한다.
    즉 RL 은 '걷는 법' 대신 '어디로 틀지(조향)'와 미세 조정만 배우면 되므로
    추격/회피가 훨씬 빨리 수렴한다.

수식:
    실제 액션 = clip( gait(t) + residual_scale * RL_residual , -1, 1 )
    관측      = 원본 관측 + [sin φ, cos φ]   (φ = 2π·t / period, 보행 위상)
        보행 위상을 관측에 넣어, RL 이 다리 주기에 맞춰 잔차를 줄 수 있게 한다.

사용:
    학습:   train.py --residual ...        (ResidualGaitEnv 로 학습)
    추론:   load_run 이 meta['residual'] 을 보고 ResidualGaitPolicy 로 로드
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def forward_gait(t: int, period: float = 20.0) -> np.ndarray:
    """검증된 직진 보행 패턴(8차원 토크). 제자리가 아니라 실제로 전진한다.

    play.py 로 단독 확인 시 토르소가 실제로 이동하는 것을 측정해 검증한 패턴.
    """
    a = np.zeros(8, dtype=np.float32)
    ph = float(np.sin(2.0 * np.pi * t / period))
    cph = float(np.cos(2.0 * np.pi * t / period))
    for k in range(4):
        sign = 1.0 if k % 2 == 0 else -1.0
        a[k * 2] = sign * ph                       # hip (대각 다리쌍 교대)
        a[k * 2 + 1] = 0.8 * np.sign(cph) * sign   # ankle (디딤/들어올림)
    return a


def phase_features(t: int, period: float) -> np.ndarray:
    """보행 위상(sin, cos) 2차원 — 관측에 덧붙여 RL 이 주기를 인지하게 한다."""
    ph = 2.0 * np.pi * t / period
    return np.array([np.sin(ph), np.cos(ph)], dtype=np.float32)


class ResidualGaitEnv(gym.Wrapper):
    """학습용 래퍼: gait 기반 + RL 잔차. 관측에 보행 위상(2D)을 추가한다.

    - 행동공간: 원본과 동일한 Box(-1,1,8) 이지만 의미는 '잔차'.
    - 관측공간: 원본 + 2(sin φ, cos φ).
    """

    def __init__(self, env: gym.Env, residual_scale: float = 0.5, period: float = 20.0):
        super().__init__(env)
        self.residual_scale = float(residual_scale)
        self.period = float(period)
        self._t = 0
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([low, [-1.0, -1.0]]).astype(np.float32),
            high=np.concatenate([high, [1.0, 1.0]]).astype(np.float32),
            dtype=np.float32,
        )

    def set_opponent(self, opponent_policy) -> None:
        """self-play 상대 스냅샷 교체를 내부 환경에 위임(gym.Wrapper 는 자동전달 안 함)."""
        self.env.set_opponent(opponent_policy)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float32),
                               phase_features(self._t, self.period)]).astype(np.float32)

    def reset(self, **kwargs):
        self._t = 0
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, residual):
        base = forward_gait(self._t, self.period)
        action = np.clip(base + self.residual_scale * np.asarray(residual, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        return self._augment(obs), reward, terminated, truncated, info


class ResidualGaitPolicy:
    """추론용: 저장된 잔차 모델을 gait 와 결합해 실제 8D 행동을 만든다.

    play.py / evaluate.py 의 policy(obs) 인터페이스(원본 obs 입력)와 호환.
    내부에서 보행 위상을 붙여 모델에 전달하고, 에피소드마다 reset() 으로 위상을 0 으로.
    """

    def __init__(self, model, residual_scale: float = 0.5, period: float = 20.0,
                 deterministic: bool = True):
        self.model = model
        self.residual_scale = float(residual_scale)
        self.period = float(period)
        self.deterministic = deterministic
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        aug = np.concatenate([np.asarray(obs, dtype=np.float32),
                              phase_features(self.t, self.period)]).astype(np.float32)
        residual, _ = self.model.predict(aug, deterministic=self.deterministic)
        action = np.clip(forward_gait(self.t, self.period)
                         + self.residual_scale * np.asarray(residual, dtype=np.float32), -1.0, 1.0)
        self.t += 1
        return np.asarray(action, dtype=np.float32)
