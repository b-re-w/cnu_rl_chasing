from agents.ppo_agent import make_ppo
from agents.sac_agent import make_sac
from agents.self_play import (
    RandomPolicy,
    SB3Policy,
    SingleAgentChaseEnv,
    StaticPolicy,
)

ALGOS = {'ppo': make_ppo, 'sac': make_sac}

# 학습 구성 프리셋: preset -> (chaser_algo, evader_algo)
PRESETS = {
    'ppo': ("ppo", "ppo"),  # ppo: 대칭 PPO
    'sac': ("sac", "sac"),  # sac: 대칭 SAC
    'asym': ("ppo", "sac"),  # asym: 비대칭 self-play (제안 알고리즘) — Chaser=PPO, Evader=SAC
}


def resolve_algos(preset: str, chaser_algo: str | None = None, evader_algo: str | None = None) -> tuple[str, str]:
    base_c, base_e = PRESETS[preset]
    return (chaser_algo or base_c, evader_algo or base_e)


__all__ = [
    "make_ppo",
    "make_sac",
    "ALGOS",
    "PRESETS",
    "resolve_algos",
    "SingleAgentChaseEnv",
    "RandomPolicy",
    "StaticPolicy",
    "SB3Policy",
]
