"""
MuJoCo 인터랙티브 뷰어로 추격-도망 시뮬레이션을 실시간 관전

학습 전이라도 환경이 동작하는 모습(아레나/두 마리 개/충전소/배터리)을
바로 확인할 수 있다. 모델을 지정하면 학습된 정책으로 플레이

사용 예:
    python play.py                              # 무작위 정책으로 즉시 관전
    python play.py --run-dir runs/asym_ppo_sac  # 학습된 정책 관전
    python play.py --chaser runs/ppo/chaser_final.zip --evader ... --preset ppo

조작:
    마우스 드래그=회전, 휠=줌, 우클릭 드래그=이동, [Space]=일시정지(뷰어 기본)
종료: 뷰어 창을 닫으면 종료
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from agents import resolve_algos
from agents.self_play import RandomPolicy
from envs.ant_chase_env import AntChaseEnv, ChaseConfig
from evaluate import load_policy, load_run


def main() -> None:
    p = argparse.ArgumentParser(description="추격-도망 실시간 뷰어")
    p.add_argument("--run-dir", default=None, help="학습 산출물 디렉터리(meta.json 자동 인식)")
    p.add_argument("--preset", default="ppo", choices=["ppo", "sac", "asym"])
    p.add_argument("--chaser-algo", default=None, choices=["ppo", "sac"])
    p.add_argument("--evader-algo", default=None, choices=["ppo", "sac"])
    p.add_argument("--chaser", default=None, help="추격자 모델 .zip")
    p.add_argument("--evader", default=None, help="도망자 모델 .zip")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율(1.0=실시간)")
    args = p.parse_args()

    if args.run_dir:
        chaser_pol, evader_pol, meta = load_run(args.run_dir)
        print(f"[play] {args.run_dir} chaser={meta['chaser_algo']} evader={meta['evader_algo']}")
    else:
        chaser_algo, evader_algo = resolve_algos(args.preset, args.chaser_algo, args.evader_algo)
        chaser_pol = load_policy(args.chaser, chaser_algo) if args.chaser else RandomPolicy()
        evader_pol = load_policy(args.evader, evader_algo) if args.evader else RandomPolicy()
        tag = "무작위" if not (args.chaser or args.evader) else "학습된"
        print(f"[play] {tag} 정책으로 관전합니다. (창을 닫으면 종료)")

    cfg = ChaseConfig(max_steps=args.max_steps)
    env = AntChaseEnv(config=cfg)

    def reset_policies():
        for pol in (chaser_pol, evader_pol):
            if hasattr(pol, "reset"):
                pol.reset()

    obs = env.reset(seed=args.seed)
    reset_policies()

    dt = env.model.opt.timestep * cfg.frame_skip
    episode = 1

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 시작 시 위에서 내려다보는 시점
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -60
        viewer.cam.distance = 28
        viewer.cam.lookat[:] = [0, 0, 0]

        while viewer.is_running():
            t0 = time.time()
            actions = {
                'chaser': np.asarray(chaser_pol(obs['chaser']), dtype=np.float32),
                'evader': np.asarray(evader_pol(obs['evader']), dtype=np.float32),
            }
            obs, rew, term, trunc, info = env.step(actions)
            viewer.sync()

            if any(term.values()) or any(trunc.values()):
                if info['evader']['caught']:
                    outcome = "포획"
                elif info['evader']['depleted']:
                    outcome = "방전"
                else:
                    outcome = "시간초과"
                print(f"[ep {episode}] {outcome} | battery={info['evader']['battery']:.2f} dist={info['evader']['distance']:.1f}")
                episode += 1
                obs = env.reset()
                reset_policies()

            # 실시간 동기화
            elapsed = time.time() - t0
            sleep = dt / max(args.speed, 1e-6) - elapsed
            if sleep > 0:
                time.sleep(sleep)

    env.close()
    print("[play] 종료")


if __name__ == "__main__":
    main()
