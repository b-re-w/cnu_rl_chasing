"""
AntChaseEnv 를 PettingZoo ParallelEnv 로 래핑

PettingZoo Parallel API:
    reset(seed, options) -> (observations, infos)
    step(actions) -> (observations, rewards, terminations, truncations, infos)
"""

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from envs.ant_chase_env import AGENTS, AntChaseEnv, ChaseConfig


class AntChaseParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel 인터페이스의 추격-도망 환경
    """

    metadata = {'render_modes': ["rgb_array"], 'name': "ant_chase_v0"}

    def __init__(self, config: ChaseConfig | None = None, render_mode: str | None = None):
        self.env = AntChaseEnv(config=config, render_mode=render_mode)
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.agents = list(AGENTS)

        self._obs_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(self.env.obs_dim[a],), dtype=np.float32)
            for a in AGENTS
        }
        self._act_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(self.env.act_dim[a],), dtype=np.float32)
            for a in AGENTS
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._act_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs = self.env.reset(seed=seed, options=options)
        self.agents = list(AGENTS)
        infos = {a: {} for a in AGENTS}
        return obs, infos

    def step(self, actions: dict):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        # 에피소드 종료 시 PettingZoo 규약상 agents 를 비운다
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def raw_parallel_env(config: ChaseConfig | None = None, render_mode: str | None = None) -> AntChaseParallelEnv:
    """
    래퍼 없는 순수 ParallelEnv 를 반환
    """
    return AntChaseParallelEnv(config=config, render_mode=render_mode)


def make_parallel_env(config: ChaseConfig | None = None, render_mode: str | None = None):
    """
    PettingZoo 표준 검증 래퍼를 씌운 AEC 환경을 반환

    `pettingzoo.test.parallel_api_test` 와는 별개로, AEC 변환 + 순서/경고 래퍼를 적용 (디버깅)
    """
    env = AntChaseParallelEnv(config=config, render_mode=render_mode)
    aec = parallel_to_aec(env)
    aec = wrappers.OrderEnforcingWrapper(aec)
    return aec
