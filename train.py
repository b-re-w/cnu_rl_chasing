"""
Self-play 공진화 학습 스크립트 (PPO / SAC / 비대칭)

추격자(chaser)와 도망자(evader)를 각각 독립된 SB3 모델로 학습하되,
상대는 직전 라운드의 정책 스냅샷으로 고정한다. 라운드를 번갈아 진행하며
두 정책이 서로를 능가하려 진화(arms race)하도록 유도

세 가지 학습 구성을 `--preset` 으로 선택:
    ppo  : 대칭 PPO  (chaser=PPO,  evader=PPO)
    sac  : 대칭 SAC  (chaser=SAC,  evader=SAC)
    asym : 비대칭 self-play(제안) (chaser=PPO, evader=SAC)
역할별 알고리즘은 --chaser-algo / --evader-algo 로 직접 지정 가능

사용 예:
    python train.py --preset ppo  --rounds 10 --steps-per-round 50000
    python train.py --preset asym --rounds 10 --steps-per-round 50000 --device cuda

산출물(--output, 기본 runs/):
    runs/<tag>/chaser_final.zip, evader_final.zip
    runs/<tag>/checkpoints/...
    runs/<tag>/history.json   (라운드별 평가 지표)
    runs/<tag>/meta.json      (역할별 알고리즘 등 구성 정보)
    runs/<tag>/tb/            (TensorBoard 로그)
"""

import argparse
import json
import os
import time

from agents import ALGOS, resolve_algos
from agents.self_play import RandomPolicy, SB3Policy, SingleAgentChaseEnv
from agents.residual import ResidualGaitEnv, ResidualGaitPolicy
from envs.ant_chase_env import ChaseConfig
from evaluate import evaluate_match


def _make_learner_env(learner: str, cfg: ChaseConfig, residual: bool = False,
                      residual_scale: float = 0.5):
    """학습 대상용 단일 에이전트 환경(초기 상대=무작위).

    residual=True 면 스크립트 보행 기반 + RL 잔차 래퍼(ResidualGaitEnv)로 감싼다.
    """
    env = SingleAgentChaseEnv(learner=learner, opponent_policy=RandomPolicy(), config=cfg)
    if residual:
        env = ResidualGaitEnv(env, residual_scale=residual_scale)
    return env


def _run_tag(preset: str, chaser_algo: str, evader_algo: str) -> str:
    """산출물 디렉터리 태그. 대칭이면 알고리즘명, 비대칭이면 c_e 형태."""
    if chaser_algo == evader_algo:
        return chaser_algo
    return f"{preset}_{chaser_algo}_{evader_algo}"


def train(
    chaser_algo: str, evader_algo: str, rounds: int, steps_per_round: int,
    output: str, tag: str, seed: int, device: str, max_steps: int,
    eval_episodes: int, warmup_random: bool, init_from: str | None = None,
    residual: bool = False, residual_scale: float = 0.5,
) -> dict:
    make_chaser = ALGOS[chaser_algo]
    make_evader = ALGOS[evader_algo]
    out_dir  = os.path.join(output, tag)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    tb_dir   = os.path.join(out_dir, "tb")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 스냅샷/평가용 상대 정책: residual 모델은 gait 와 결합해야 하므로 래핑이 다르다
    def snap(model):
        return ResidualGaitPolicy(model, residual_scale=residual_scale) if residual else SB3Policy(model)

    # 구성 정보 저장(평가/시각화 시 자동 로드용)
    meta = {
        'chaser_algo': chaser_algo, 'evader_algo': evader_algo, 'tag': tag,
        'rounds': rounds, 'steps_per_round': steps_per_round, 'max_steps': max_steps,
        'residual': residual, 'residual_scale': residual_scale,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    cfg = ChaseConfig(max_steps=max_steps)

    # 학습 대상 환경(각자 자신을 학습, 상대는 스냅샷)
    chaser_env = _make_learner_env("chaser", cfg, residual=residual, residual_scale=residual_scale)
    evader_env = _make_learner_env("evader", cfg, residual=residual, residual_scale=residual_scale)

    chaser = make_chaser(chaser_env, seed=seed, tensorboard_log=tb_dir, device=device)
    evader = make_evader(evader_env, seed=seed + 1, tensorboard_log=tb_dir, device=device)

    # 워밍스타트: 기존 산출물의 가중치(정책+탐험노이즈)를 이어받아 시작
    if init_from:
        chaser.set_parameters(os.path.join(init_from, "chaser_final.zip"), device=device)
        evader.set_parameters(os.path.join(init_from, "evader_final.zip"), device=device)
        print(f"[train] 워밍스타트: {init_from} 의 가중치를 이어받아 시작")

    history = []
    print(f"[train] tag={tag} chaser={chaser_algo} evader={evader_algo} rounds={rounds} steps/round={steps_per_round} device={device}")

    for rnd in range(1, rounds + 1):
        t0 = time.time()

        # --- 상대 스냅샷 설정 ---
        if warmup_random and rnd == 1:
            chaser_env.set_opponent(RandomPolicy())
            evader_env.set_opponent(RandomPolicy())
        else:
            # 추격자는 현재 도망자를, 도망자는 현재 추격자를 상대로
            chaser_env.set_opponent(snap(evader))
            evader_env.set_opponent(snap(chaser))

        # --- 추격자 학습 ---
        chaser.learn(total_timesteps=steps_per_round, reset_num_timesteps=False, progress_bar=True, tb_log_name="chaser")

        # 도망자 학습 시에는 방금 갱신된 추격자를 상대로
        evader_env.set_opponent(snap(chaser))
        evader.learn(total_timesteps=steps_per_round, reset_num_timesteps=False, progress_bar=True, tb_log_name="evader")

        # --- 평가(학습된 두 정책 맞대결) ---
        metrics = evaluate_match(snap(chaser), snap(evader), config=cfg, episodes=eval_episodes, seed=10_000 + rnd)
        metrics['round'] = rnd
        metrics['seconds'] = round(time.time() - t0, 1)
        history.append(metrics)

        print(
            f"[round {rnd:>2}/{rounds}] "
            f"catch={metrics['catch_rate']:.2f} "
            f"charge={metrics['charge_success_rate']:.2f} "
            f"survive={metrics['mean_survival_steps']:.0f} "
            f"R_c={metrics['mean_return_chaser']:.1f} "
            f"R_e={metrics['mean_return_evader']:.1f} "
            f"({metrics['seconds']}s)"
        )

        # --- 체크포인트 ---
        chaser.save(os.path.join(ckpt_dir, f"chaser_r{rnd}"))
        evader.save(os.path.join(ckpt_dir, f"evader_r{rnd}"))
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    # --- 최종 모델 저장 ---
    chaser.save(os.path.join(out_dir, "chaser_final"))
    evader.save(os.path.join(out_dir, "evader_final"))
    chaser_env.close()
    evader_env.close()
    print(f"[train] 완료. 모델 저장 → {out_dir}/")
    return {'history': history, 'output': out_dir, 'chaser_algo': chaser_algo, 'evader_algo': evader_algo}


def main() -> None:
    p = argparse.ArgumentParser(description="Self-play 추격-도망 학습")
    p.add_argument("--preset", default="ppo", choices=["ppo", "sac", "asym"], help="학습 구성 프리셋 (ppo/sac/asym)")
    p.add_argument("--chaser-algo", default=None, choices=["ppo", "sac"], help="추격자 알고리즘(프리셋 덮어쓰기)")
    p.add_argument("--evader-algo", default=None, choices=["ppo", "sac"], help="도망자 알고리즘(프리셋 덮어쓰기)")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--steps-per-round", type=int, default=50_000)
    p.add_argument("--output", default="runs")
    p.add_argument("--tag", default=None, help="산출물 디렉터리 이름(기본 자동)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--no-warmup-random", action="store_true", help="1라운드부터 스냅샷 상대 사용")
    p.add_argument("--init-from", default=None, help="기존 산출물 디렉터리에서 가중치를 이어받아 시작(워밍스타트)")
    p.add_argument("--residual", action="store_true",
                   help="스크립트 보행 기반 + RL 조향 잔차 방식으로 학습(걷기 학습 불필요)")
    p.add_argument("--residual-scale", type=float, default=0.5, help="잔차 액션 스케일(기본 0.5)")
    args = p.parse_args()

    # residual 은 관측에 보행 위상 2D 를 더해 차원이 달라지므로, 비-residual 워밍스타트와 호환 불가
    if args.residual and args.init_from:
        p.error("--residual 과 --init-from 은 함께 쓸 수 없습니다(관측 차원 불일치). "
                "residual 은 스크립트 보행이 운동 prior 라 워밍스타트가 필요 없습니다.")

    chaser_algo, evader_algo = resolve_algos(args.preset, args.chaser_algo, args.evader_algo)
    tag = args.tag or _run_tag(args.preset, chaser_algo, evader_algo)

    train(
        chaser_algo=chaser_algo, evader_algo=evader_algo,
        rounds=args.rounds, steps_per_round=args.steps_per_round,
        output=args.output, tag=tag, seed=args.seed, device=args.device,
        max_steps=args.max_steps, eval_episodes=args.eval_episodes,
        warmup_random=not args.no_warmup_random, init_from=args.init_from,
        residual=args.residual, residual_scale=args.residual_scale,
    )


if __name__ == "__main__":
    main()
