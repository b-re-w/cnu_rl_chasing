"""
Self-play 학습을 위한 단일 에이전트 Gymnasium 래퍼

Stable-Baselines3 는 단일 에이전트 환경만 지원하므로, 멀티에이전트
AntChaseEnv 를 "학습 대상 1명 + 고정 상대 1명" 관점으로 노출

- `learner`  : SB3 가 학습하는 에이전트("chaser" 또는 "evader").
- `opponent` : 고정된 정책(스냅샷)으로 행동하는 상대

self-play 루프(train.py)에서 상대 정책을 주기적으로 최신 스냅샷으로
교체하며 공진화(arms race)를 유도
"""

from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.ant_chase_env import AGENTS, AntChaseEnv, ChaseConfig

# 상대 정책: 상대의 관측(np.ndarray) -> 행동(np.ndarray, shape (8,))
OpponentPolicy = Callable[[np.ndarray], np.ndarray]


class RandomPolicy:
    """
    무작위 행동 상대 (학습 초기 부트스트랩)
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=8).astype(np.float32)


class StaticPolicy:
    """
    아무 행동도 하지 않는 상대 (정지)
    """

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


class SB3Policy:
    """
    학습된 SB3 모델을 상대 정책으로 랩핑 (결정적 추론)
    """

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


class SingleAgentChaseEnv(gym.Env):
    """
    AntChaseEnv 를 한 에이전트 관점의 단일 에이전트 환경으로 변환

    Args:
        learner:          학습 대상 ("chaser" | "evader").
        opponent_policy:  상대 행동을 결정하는 콜러블. None 이면 정지.
        config:           ChaseConfig.
        render_mode:      "rgb_array" 등.
    """

    metadata = {'render_modes': ["rgb_array"], 'render_fps': 20}

    def __init__(
        self, learner: str = "chaser", opponent_policy: OpponentPolicy | None = None,
        config: ChaseConfig | None = None, render_mode: str | None = None,
    ):
        assert learner in AGENTS, f"learner must be one of {AGENTS}"
        super().__init__()
        self.learner = learner
        self.opponent = "evader" if learner == "chaser" else "chaser"
        self.opponent_policy = opponent_policy or StaticPolicy()
        self.env = AntChaseEnv(config=config, render_mode=render_mode)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.env.obs_dim[self.learner],), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.env.act_dim[self.learner],), dtype=np.float32)

        self._last_obs: dict | None = None

    def set_opponent(self, opponent_policy: OpponentPolicy) -> None:
        """
        상대 정책 스냅샷을 교체 (self-play 라운드 전환)
        """
        self.opponent_policy = opponent_policy

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs[self.learner], {}

    def step(self, action):
        opp_obs = self._last_obs[self.opponent]
        opp_action = self.opponent_policy(opp_obs)
        actions = {
            self.learner: np.asarray(action, dtype=np.float32),
            self.opponent: np.asarray(opp_action, dtype=np.float32),
        }

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_obs = obs

        return (
            obs[self.learner],
            float(rewards[self.learner]),
            bool(terminations[self.learner]),
            bool(truncations[self.learner]),
            infos[self.learner],
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
