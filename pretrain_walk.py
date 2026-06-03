"""
걷기(locomotion) 사전학습 — 추격/회피 학습 전에 "걷는 법"부터 습득

문제의식:
    Ant 는 (1) 걷기(저수준 운동)와 (2) 추격/회피(고수준 전략)를 동시에 백지에서
    배워야 해서 학습이 매우 느리다. 걷지 못하면 보상 신호가 약해 "제자리 정지"
    지역최적해에 빠진다.

해결:
    상대도 배터리도 없는 단일 에이전트 환경에서 "전방으로 빨리 걷기"만 보상으로
    주어 보행을 먼저 습득시킨다. 같은 몸(morphology)·같은 관측(observation)을
    쓰므로 가중치가 100% 호환된다. 학습된 walker 를 train.py 의 --init-from 으로
    이어받으면 추격/회피 학습이 훨씬 빨라진다.

    표준 Gymnasium Ant 의 사전학습 가중치는 관측 차원(우리 38/41 vs 표준 27)과
    몸 구조가 달라 그대로 쓸 수 없으므로, 우리 env 에서 직접 사전학습한다.

산출물:
    runs/walk/chaser_final.zip, runs/walk/evader_final.zip, meta.json
    (그대로 play.py / evaluate.py 로 걷는 모습 확인 가능)

사용 예:
    python pretrain_walk.py --steps 400000
    python train.py --preset ppo --rounds 12 --steps-per-round 100000 --init-from runs/walk --tag ppo_v2
"""

import argparse
import json
import os

import numpy as np
import gymnasium as gym

from agents import ALGOS
from agents.self_play import SingleAgentChaseEnv, StaticPolicy
from envs.ant_chase_env import AGENTS, ChaseConfig, planar_forward_dir


class WalkRewardWrapper(gym.Wrapper):
    """
    SingleAgentChaseEnv 의 보상을 '전방 속도 기반 보행 보상'으로 교체

    관측 벡터 앞부분 레이아웃(own state):
        [0:2]=xy, [2]=z(높이), [3:7]=quat[w,x,y,z], [7:10]=linvel, ...

    보상 = forward_coef * (전방 속도) + healthy_bonus - ctrl_cost * Σa²
        - 전방 속도는 본체가 '바라보는 방향'으로의 속도(부호 있음)라
          제자리 진동(앞뒤 떨기)으로 farming 되지 않는다.
        - healthy_bonus 로 넘어지지 않고 서 있도록 유도.
    """

    def __init__(self, env, forward_coef: float = 1.0, healthy_bonus: float = 0.05,
                 ctrl_cost: float = 0.0005):
        super().__init__(env)
        self.forward_coef = forward_coef
        self.healthy_bonus = healthy_bonus
        self.ctrl_cost = ctrl_cost

    def step(self, action):
        obs, _orig_r, terminated, truncated, info = self.env.step(action)
        vel_xy = obs[7:9]
        quat = obs[3:7]
        fwd_speed = float(vel_xy @ planar_forward_dir(quat))
        ctrl = float(np.sum(np.square(action)))
        reward = self.forward_coef * fwd_speed + self.healthy_bonus - self.ctrl_cost * ctrl
        return obs, reward, terminated, truncated, info


def make_walk_env(learner: str, cfg: ChaseConfig) -> gym.Env:
    """걷기 전용 단일 에이전트 환경(상대=정지, 포획 판정 비활성)."""
    # catch_radius=0 으로 포획 종료를 막아 보행에 집중(넘어짐만 종료로 남김)
    base = SingleAgentChaseEnv(learner=learner, opponent_policy=StaticPolicy(), config=cfg)
    return WalkRewardWrapper(base)


def pretrain(algo: str, steps: int, output: str, tag: str, seed: int, device: str,
             max_steps: int) -> str:
    out_dir = os.path.join(output, tag)
    os.makedirs(out_dir, exist_ok=True)
    # 보행은 포획/배터리와 무관 → catch_radius=0 으로 조기 종료 방지
    cfg = ChaseConfig(max_steps=max_steps, catch_radius=0.0)

    make = ALGOS[algo]
    for i, learner in enumerate(AGENTS):
        env = make_walk_env(learner, cfg)
        model = make(env, seed=seed + i, tensorboard_log=os.path.join(out_dir, "tb"), device=device)
        print(f"[pretrain] {learner} walker 학습 시작 (obs={env.observation_space.shape[0]}, {steps} steps)")
        model.learn(total_timesteps=steps, progress_bar=True)
        model.save(os.path.join(out_dir, f"{learner}_final"))
        env.close()
        print(f"[pretrain] {learner} walker 저장 → {out_dir}/{learner}_final.zip")

    # play.py / evaluate.py 의 load_run 호환용 meta
    meta = {'chaser_algo': algo, 'evader_algo': algo, 'tag': tag,
            'rounds': 0, 'steps_per_round': steps, 'max_steps': max_steps,
            'pretrain': 'walk'}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[pretrain] 완료. --init-from {out_dir} 로 추격/회피 학습을 이어받으세요.")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="걷기(locomotion) 사전학습")
    p.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    p.add_argument("--steps", type=int, default=400_000, help="에이전트당 학습 스텝")
    p.add_argument("--output", default="runs")
    p.add_argument("--tag", default="walk", help="산출물 디렉터리 이름")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-steps", type=int, default=1000)
    args = p.parse_args()

    pretrain(algo=args.algo, steps=args.steps, output=args.output, tag=args.tag,
             seed=args.seed, device=args.device, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
