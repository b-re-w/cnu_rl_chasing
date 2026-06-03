"""
학습된 정책의 추격-도망 에피소드를 영상(mp4/gif)으로 저장

사용 예:
    python visualize.py --algo ppo \
        --chaser runs/ppo/chaser_final.zip \
        --evader runs/ppo/evader_final.zip \
        --episodes 3 --out videos/ppo_match.mp4

모델 경로를 생략하면 해당 에이전트는 무작위 정책으로 대체
"""

import argparse
import os

import numpy as np
import imageio.v2 as imageio

from agents import resolve_algos
from envs.ant_chase_env import AntChaseEnv, ChaseConfig
from evaluate import load_policy, load_run, run_episode


def main() -> None:
    p = argparse.ArgumentParser(description="추격-도망 에피소드 영상 저장")
    p.add_argument("--run-dir", default=None, help="학습 산출물 디렉터리(meta.json 으로 알고리즘 자동 인식)")
    p.add_argument("--preset", default="ppo", choices=["ppo", "sac", "asym"])
    p.add_argument("--chaser-algo", default=None, choices=["ppo", "sac"])
    p.add_argument("--evader-algo", default=None, choices=["ppo", "sac"])
    p.add_argument("--chaser", default=None)
    p.add_argument("--evader", default=None)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--camera", default="topdown")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out", default="videos/match.mp4")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.run_dir:
        chaser_pol, evader_pol, meta = load_run(args.run_dir)
        print(f"[viz] run={args.run_dir} chaser={meta['chaser_algo']} evader={meta['evader_algo']}")
    else:
        chaser_algo, evader_algo = resolve_algos(args.preset, args.chaser_algo, args.evader_algo)
        chaser_pol = load_policy(args.chaser, chaser_algo)
        evader_pol = load_policy(args.evader, evader_algo)

    cfg = ChaseConfig(max_steps=args.max_steps)
    env = AntChaseEnv(config=cfg, render_mode="rgb_array")

    all_frames = []
    for ep in range(args.episodes):
        result = run_episode(env, chaser_pol, evader_pol, seed=args.seed + ep, render=True, camera=args.camera)
        all_frames.extend(result['frames'])
        outcome = "포획" if result['caught'] else "방전" if result['depleted'] else "생존"
        print(
            f"[ep {ep + 1}] {outcome} | steps={result['steps']} "
            f"battery={result['final_battery']:.2f} "
            f"R_c={result['return_chaser']:.1f} "
            f"R_e={result['return_evader']:.1f}"
        )
    env.close()

    if not all_frames:
        print("프레임이 없습니다.")
        return

    frames = [np.asarray(f, dtype=np.uint8) for f in all_frames]
    if args.out.lower().endswith(".gif"):
        imageio.mimsave(args.out, frames, fps=args.fps)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=None)
    print(f"영상 저장 완료 → {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
