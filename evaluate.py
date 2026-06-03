"""
학습된 추격자/도망자 정책의 성능 평가

비교 지표:
- 포획률(catch_rate)
- 평균 생존 시간(mean_survival_steps)
- 충전 성공률(charge_success_rate)
- 방전률(depletion_rate)
- 에이전트별 평균 누적 보상
"""

from typing import Callable
import argparse
import json
import os

import numpy as np

from agents import resolve_algos
from agents.self_play import RandomPolicy, SB3Policy, StaticPolicy
from envs.ant_chase_env import AGENTS, AntChaseEnv, ChaseConfig

Policy = Callable[[np.ndarray], np.ndarray]


def run_episode(
    env: AntChaseEnv, chaser_pol: Policy, evader_pol: Policy,
    seed: int | None = None, render: bool = False, camera: str = "topdown",
) -> dict:
    """
    한 에피소드를 실행하고 통계를 반환
    """
    obs = env.reset(seed=seed)
    # residual 정책 등 상태가 있는 정책은 에피소드마다 보행 위상을 초기화
    for pol in (chaser_pol, evader_pol):
        if hasattr(pol, "reset"):
            pol.reset()
    ret = {a: 0.0 for a in AGENTS}
    frames = []
    charged = False
    steps = 0
    info = {'evader': {'caught': False, 'depleted': False, 'battery': 1.0}}

    while True:
        actions = {
            'chaser': np.asarray(chaser_pol(obs['chaser']), dtype=np.float32),
            'evader': np.asarray(evader_pol(obs['evader']), dtype=np.float32),
        }
        obs, rew, term, trunc, info = env.step(actions)
        for a in AGENTS:
            ret[a] += rew[a]
        steps += 1
        if info['evader']['charging']:
            charged = True
        if render:
            frames.append(env.render(camera))
        if any(term.values()) or any(trunc.values()):
            break

    return {
        'steps': steps,
        'caught': bool(info['evader']['caught']),
        'depleted': bool(info['evader']['depleted']),
        'charged': charged,
        'final_battery': float(info['evader']['battery']),
        'return_chaser': ret['chaser'],
        'return_evader': ret['evader'],
        'frames': frames,
    }


def evaluate_match(
    chaser_pol: Policy, evader_pol: Policy, config: ChaseConfig | None = None,
    episodes: int = 20, seed: int = 1000,
) -> dict:
    """
    여러 에피소드에 걸쳐 지표를 집계
    """
    env = AntChaseEnv(config=config or ChaseConfig())
    stats = []
    for i in range(episodes):
        stats.append(run_episode(env, chaser_pol, evader_pol, seed=seed + i))
    env.close()

    def mean(key):
        return float(np.mean([s[key] for s in stats]))

    return {
        'episodes': episodes,
        'catch_rate': mean("caught"),
        'depletion_rate': mean("depleted"),
        'charge_success_rate': mean("charged"),
        'mean_survival_steps': mean("steps"),
        'mean_final_battery': mean("final_battery"),
        'mean_return_chaser': mean("return_chaser"),
        'mean_return_evader': mean("return_evader"),
    }


def load_policy(path: str | None, algo: str, kind: str = "random") -> Policy:
    """
    모델 경로로부터 정책을 만든다. path 가 없으면 기본 정책 사용
    """
    if path is None:
        return RandomPolicy() if kind == "random" else StaticPolicy()
    from stable_baselines3 import PPO, SAC
    cls = {'ppo': PPO, 'sac': SAC}[algo.lower()]
    model = cls.load(path, device="cpu")
    return SB3Policy(model)


def _load_model(path: str, algo: str):
    from stable_baselines3 import PPO, SAC
    cls = {'ppo': PPO, 'sac': SAC}[algo.lower()]
    return cls.load(path, device="cpu")


def load_run(run_dir: str) -> tuple[Policy, Policy, dict]:
    """
    학습 산출물 디렉터리에서 meta.json 을 읽어 두 정책을 로드

    meta['residual'] 이면 스크립트 보행 + RL 잔차 정책(ResidualGaitPolicy)으로 로드.

    Returns:
        (chaser_policy, evader_policy, meta)
    """
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    c_path = os.path.join(run_dir, "chaser_final.zip")
    e_path = os.path.join(run_dir, "evader_final.zip")
    if meta.get('residual'):
        from agents.residual import ResidualGaitPolicy
        scale = meta.get('residual_scale', 0.5)
        chaser = ResidualGaitPolicy(_load_model(c_path, meta['chaser_algo']), residual_scale=scale)
        evader = ResidualGaitPolicy(_load_model(e_path, meta['evader_algo']), residual_scale=scale)
    else:
        chaser = load_policy(c_path, meta['chaser_algo'])
        evader = load_policy(e_path, meta['evader_algo'])
    return chaser, evader, meta


def _print_metrics(metrics: dict) -> None:
    print("\n===== 평가 결과 =====")
    print(f"에피소드 수            : {metrics['episodes']}")
    print(f"포획률 (catch)        : {metrics['catch_rate']:.2%}")
    print(f"방전률 (depletion)    : {metrics['depletion_rate']:.2%}")
    print(f"충전 성공률 (charge)  : {metrics['charge_success_rate']:.2%}")
    print(f"평균 생존 스텝        : {metrics['mean_survival_steps']:.1f}")
    print(f"평균 최종 배터리      : {metrics['mean_final_battery']:.3f}")
    print(f"평균 보상 (chaser)    : {metrics['mean_return_chaser']:.2f}")
    print(f"평균 보상 (evader)    : {metrics['mean_return_evader']:.2f}")
    print("=====================\n")


def main() -> None:
    p = argparse.ArgumentParser(description="추격-도망 정책 평가")
    p.add_argument("--run-dir", default=None, help="학습 산출물 디렉터리(meta.json 으로 알고리즘 자동 인식)")
    p.add_argument("--preset", default="ppo", choices=["ppo", "sac", "asym"])
    p.add_argument("--chaser-algo", default=None, choices=["ppo", "sac"])
    p.add_argument("--evader-algo", default=None, choices=["ppo", "sac"])
    p.add_argument("--chaser", default=None, help="추격자 모델 .zip 경로")
    p.add_argument("--evader", default=None, help="도망자 모델 .zip 경로")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    if args.run_dir:
        chaser_pol, evader_pol, meta = load_run(args.run_dir)
        print(f"[eval] run={args.run_dir} chaser={meta['chaser_algo']} evader={meta['evader_algo']}")
    else:
        chaser_algo, evader_algo = resolve_algos(args.preset, args.chaser_algo, args.evader_algo)
        chaser_pol = load_policy(args.chaser, chaser_algo)
        evader_pol = load_policy(args.evader, evader_algo)

    cfg = ChaseConfig(max_steps=args.max_steps)
    metrics = evaluate_match(chaser_pol, evader_pol, config=cfg, episodes=args.episodes, seed=args.seed)
    _print_metrics(metrics)


if __name__ == "__main__":
    main()
