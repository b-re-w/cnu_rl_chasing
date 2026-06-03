"""AntChaseEnv / PettingZoo 래퍼 / self-play 래퍼 스모크 테스트"""

import numpy as np

from agents.self_play import RandomPolicy, SingleAgentChaseEnv
from envs.ant_chase_env import AGENTS, AntChaseEnv, ChaseConfig
from envs.multiagent_wrapper import raw_parallel_env


def _rand_actions():
    return {a: np.random.uniform(-1, 1, 8).astype(np.float32) for a in AGENTS}


def test_core_env_shapes_and_step():
    env = AntChaseEnv(config=ChaseConfig(max_steps=30))
    obs = env.reset(seed=0)
    assert set(obs) == set(AGENTS)
    assert obs['chaser'].shape == (env.obs_dim['chaser'],)
    assert obs['evader'].shape == (env.obs_dim['evader'],)
    obs, rew, term, trunc, info = env.step(_rand_actions())
    assert set(rew) == set(AGENTS)
    assert np.isfinite(rew['chaser']) and np.isfinite(rew['evader'])
    assert 'battery' in info['evader']
    env.close()


def test_episode_truncates():
    env = AntChaseEnv(config=ChaseConfig(max_steps=10))
    env.reset(seed=1)
    done = False
    for _ in range(10):
        actions = {a: np.zeros(8, np.float32) for a in AGENTS}
        _, _, term, trunc, _ = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            done = True
            break
    assert done
    env.close()


def test_pettingzoo_wrapper():
    env = raw_parallel_env(ChaseConfig(max_steps=20))
    obs, infos = env.reset(seed=2)
    assert set(obs) == set(AGENTS)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, infos = env.step(actions)
    assert set(rew) == set(AGENTS)
    env.close()


def test_single_agent_wrapper():
    env = SingleAgentChaseEnv("chaser", opponent_policy=RandomPolicy(0), config=ChaseConfig(max_steps=15))
    obs, _ = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(rew, float)
    env.close()
