"""
커스텀 멀티에이전트 추격-도망 환경 (MuJoCo 기반)
- 두 마리의 Ant 로봇("chaser", "evader")이 한 씬에서 경쟁
- 이 클래스는 PettingZoo/Gymnasium 래핑과 무관한 "엔진" 역할을 하며,
- 딕셔너리 형태의 행동을 받아 딕셔너리 형태의 관측/보상/종료를 반환
"""

from dataclasses import dataclass

import numpy as np
import mujoco

from envs.battery_system import BatterySystem
from envs.model_builder import build_model_xml

AGENTS = ("chaser", "evader")
# 각 Ant 의 8개 힌지 관절(액추에이터 순서와 무관, 관측 일관성을 위한 고정 순서)
_HINGES = ("hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4")


def planar_forward_dir(quat: np.ndarray) -> np.ndarray:
    """본체 쿼터니언([w,x,y,z])에서 평면 전방 단위 방향벡터(2D)를 구한다.

    보행 보상을 '전방 속도'로 계산할 때 사용. 전방 속도는 부호가 있어
    제자리 진동(앞뒤로 떨기)은 +/− 가 상쇄돼 farming 되지 않는다.
    """
    w, x, y, z = quat
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    n = float(np.hypot(fx, fy)) + 1e-8
    return np.array([fx / n, fy / n])


@dataclass
class ChaseConfig:
    """
    환경 하이퍼파라미터 (보상 계수·물리·종료 조건)
    """

    # --- 에피소드/물리 ---
    max_steps: int = 1000          # 에피소드 최대 스텝(truncation)
    frame_skip: int = 5            # 행동 1회당 물리 서브스텝 수
    arena_size: float = 10.0       # 아레나 반경
    catch_radius: float = 1.0      # 포획 판정 거리
    reset_noise: float = 0.1       # reset 시 관절 위치/속도 노이즈
    min_spawn_sep: float = 4.0     # 두 에이전트 초기 최소 이격거리
    healthy_z_range: tuple = (0.2, 1.5)  # 본체 높이 정상 범위
    spawn_height: float = 0.55           # reset 시 torso 초기 높이

    # --- 충전소 ---
    charger_positions: tuple = ((6.0, 6.0), (-6.0, -6.0))
    randomize_spawn: bool = True

    # --- 장애물(아레나 내부 실제 충돌체) ---
    obstacle_positions: tuple = ((4.5, 0.0), (-4.5, 0.0), (0.0, 4.5), (0.0, -4.5))
    obstacle_radius: float = 0.5    # 충돌 반경
    obstacle_height: float = 2.0    # 기둥 전체 높이

    # --- 배터리 ---
    battery_drain_rate: float = 0.01
    battery_idle_drain: float = 0.0005
    battery_charge_rate: float = 0.05
    battery_charge_range: float = 1.5

    # --- 추격자 보상 계수 ---
    chaser_catch_bonus: float = 10.0
    chaser_move_coef: float = 0.05        # 보행 부트스트랩: 전방 속도 보상(걷기 발견 가속). 잘 걸으면 0으로 낮춰도 됨
    chaser_progress_coef: float = 1.0     # +coef * (이전거리 - 현재거리): 접근 보상(보행 유도 핵심)
    chaser_distance_coef: float = 0.01    # -coef * dist (절대 거리, 약한 전역 신호)
    chaser_block_bonus: float = 0.05      # (스텝당) 도망자 충전소 선점 — 캠핑 farming 방지로 작게+조건부
    chaser_block_range: float = 2.0
    chaser_energy_coef: float = 0.0005    # 보행 학습 전 "가만히 있기" 함정 방지를 위해 작게
    chaser_time_penalty: float = 0.005    # 빨리 잡도록 시간 패널티

    # --- 도망자 보상 계수 ---
    evader_move_coef: float = 0.05        # 보행 부트스트랩: 전방 속도 보상(추격자와 동일 취지)
    evader_progress_coef: float = 1.0     # +coef * (현재거리 - 이전거리): 이탈 보상(보행 유도 핵심)
    evader_distance_coef: float = 0.01    # +coef * dist (절대 거리, 약한 전역 신호)
    evader_charge_bonus: float = 5.0      # 충전 성공(스텝당)
    evader_battery_coef: float = 0.1      # +coef * battery_level
    evader_caught_penalty: float = 10.0
    evader_empty_penalty: float = 5.0
    evader_energy_coef: float = 0.0005    # 동일한 이유로 작게


class AntChaseEnv:
    """
    두 Ant 의 추격-도망 멀티에이전트 엔진

    딕셔너리 API:
        reset()  -> obs(dict)
        step(actions: dict) -> (obs, rewards, terminations, truncations, infos)
    각 dict 의 키는 ("chaser", "evader")
    """

    metadata = {'render_modes': ["rgb_array", "human"], 'render_fps': 20}

    def __init__(self, config: ChaseConfig | None = None, render_mode: str | None = None):
        self.cfg = config or ChaseConfig()
        self.render_mode = render_mode

        self.chargers  = np.array(self.cfg.charger_positions, dtype=np.float64)
        self.obstacles = np.array(self.cfg.obstacle_positions, dtype=np.float64).reshape(-1, 2)

        # --- MuJoCo 모델 구성 ---
        xml = build_model_xml(
            charger_positions=self.cfg.charger_positions,
            arena_size=self.cfg.arena_size,
            charge_range=self.cfg.battery_charge_range,
            obstacle_positions=self.cfg.obstacle_positions,
            obstacle_radius=self.cfg.obstacle_radius,
            obstacle_height=self.cfg.obstacle_height,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # 관절/본체 주소를 이름으로 한 번만 조회해 캐싱
        self._index_model()

        # 배터리(도망자 전용)
        self.battery = BatterySystem(
            drain_rate=self.cfg.battery_drain_rate,
            idle_drain=self.cfg.battery_idle_drain,
            charge_rate=self.cfg.battery_charge_rate,
            charge_range=self.cfg.battery_charge_range,
        )

        self.step_count = 0
        self._renderer = None

        # 관측/행동 차원 결정(실제 obs 를 한 번 만들어 길이 측정)
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        self.obs_dim = {a: obs[a].shape[0] for a in AGENTS}
        self.act_dim = {a: 8 for a in AGENTS}

    # ------------------------------------------------------------------ #
    # 모델 인덱싱
    # ------------------------------------------------------------------ #
    def _index_model(self) -> None:
        m = self.model
        self.torso_bid  = {}
        self.root_qadr  = {}
        self.root_vadr  = {}
        self.hinge_qadr = {}
        self.hinge_vadr = {}
        self.ctrl_slice = {}
        for i, agent in enumerate(AGENTS):
            p = f"{agent}_"
            self.torso_bid[agent] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p}torso")
            rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}root")
            self.root_qadr[agent] = int(m.jnt_qposadr[rid])
            self.root_vadr[agent] = int(m.jnt_dofadr[rid])
            self.hinge_qadr[agent] = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}{h}")] for h in _HINGES])
            self.hinge_vadr[agent] = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}{h}")] for h in _HINGES])
            # 액추에이터는 chaser 8개 → evader 8개 순서로 정의됨
            self.ctrl_slice[agent] = slice(i * 8, i * 8 + 8)

    # ------------------------------------------------------------------ #
    # 상태 조회 헬퍼
    # ------------------------------------------------------------------ #
    def _ant_state(self, agent: str) -> dict:
        d = self.data
        qa, va = self.root_qadr[agent], self.root_vadr[agent]
        pos = d.xpos[self.torso_bid[agent]].copy()        # 월드 좌표 torso
        quat = d.qpos[qa + 3:qa + 7].copy()               # [w,x,y,z]
        linvel = d.qvel[va:va + 3].copy()
        angvel = d.qvel[va + 3:va + 6].copy()
        jpos = d.qpos[self.hinge_qadr[agent]].copy()
        jvel = d.qvel[self.hinge_vadr[agent]].copy()
        return {
            'pos': pos, 'quat': quat, 'linvel': linvel, 'angvel': angvel,
            'jpos': jpos, 'jvel': jvel, 'speed': float(np.linalg.norm(linvel[:2])),
        }

    def _nearest_charger(self, xy: np.ndarray) -> tuple:
        diffs = self.chargers - xy[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx]), self.chargers[idx]

    def _obstacle_obs(self, xy: np.ndarray) -> list:
        """
        가장 가까운 장애물의 상대 위치(dx, dy)와 거리
        장애물 없으면 []
        """
        if self.obstacles.shape[0] == 0:
            return []
        diffs = self.obstacles - xy[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        rel = diffs[idx]
        return [float(rel[0]), float(rel[1]), float(dists[idx])]

    # ------------------------------------------------------------------ #
    # 관측 구성
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> dict:
        cs = self._ant_state("chaser")
        es = self._ant_state("evader")

        c_xy, e_xy = cs['pos'][:2], es['pos'][:2]
        c_to_e = e_xy - c_xy
        dist_ce = float(np.linalg.norm(c_to_e))

        _, c_charge_dist, c_charger = self._nearest_charger(c_xy)
        e_charge_idx, e_charge_dist, e_charger = self._nearest_charger(e_xy)

        def own(s):
            return np.concatenate([
                s['pos'][:2], s['pos'][2:3], s['quat'],
                s['linvel'], s['angvel'], s['jpos'], s['jvel'],
            ])

        # 추격자 관측
        chaser_obs = np.concatenate([
            own(cs),
            c_to_e, [dist_ce],                       # 도망자 상대 위치/거리
            c_charger - c_xy, [c_charge_dist],       # (도망자가 노릴) 충전소 정보
            self._obstacle_obs(c_xy),                # 가장 가까운 장애물 상대/거리
        ]).astype(np.float32)

        # 도망자 관측(+배터리, +충전소 방향)
        e_to_charger = e_charger - e_xy
        charger_dir = e_to_charger / (e_charge_dist + 1e-8)
        evader_obs = np.concatenate([
            own(es),
            c_xy - e_xy, [dist_ce],                  # 추격자 상대 위치/거리
            e_to_charger, [e_charge_dist],           # 충전소 상대 위치/거리
            [self.battery.normalized],               # 배터리 잔량
            charger_dir,                             # 충전소 방향 단위벡터
            self._obstacle_obs(e_xy),                # 가장 가까운 장애물 상대/거리
        ]).astype(np.float32)

        return {'chaser': chaser_obs, 'evader': evader_obs}

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, options: dict | None = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()

        mujoco.mj_resetData(self.model, self.data)

        # 초기 위치 설정
        if self.cfg.randomize_spawn:
            c_xy, e_xy = self._sample_spawns()
        else:
            c_xy = np.array([-3.0, 0.0])
            e_xy = np.array([3.0, 0.0])

        for agent, xy in (("chaser", c_xy), ("evader", e_xy)):
            qa = self.root_qadr[agent]
            self.data.qpos[qa:qa + 2] = xy
            self.data.qpos[qa + 2] = self.cfg.spawn_height   # 높이
            self.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]  # 기본 자세
            # 관절 노이즈
            n = self.cfg.reset_noise
            self.data.qpos[self.hinge_qadr[agent]] += self._rng.uniform(-n, n, size=8)
            self.data.qvel[self.hinge_vadr[agent]] += self._rng.uniform(-n, n, size=8)

        mujoco.mj_forward(self.model, self.data)

        self.battery.reset()
        self.step_count = 0
        # progress shaping 기준점: 초기 두 에이전트 간 거리
        cs = self._ant_state("chaser")
        es = self._ant_state("evader")
        self._prev_dist_ce = float(np.linalg.norm(es['pos'][:2] - cs['pos'][:2]))
        return self._get_obs()

    def _clear_of_obstacles(self, xy: np.ndarray) -> bool:
        if self.obstacles.shape[0] == 0:
            return True
        clearance = self.cfg.obstacle_radius + 1.0
        return bool(np.all(np.linalg.norm(self.obstacles - xy[None, :], axis=1) >= clearance))

    def _sample_spawns(self) -> tuple:
        lim = self.cfg.arena_size - 1.5
        for _ in range(200):
            c = self._rng.uniform(-lim, lim, size=2)
            e = self._rng.uniform(-lim, lim, size=2)
            sep_ok = np.linalg.norm(c - e) >= self.cfg.min_spawn_sep
            if sep_ok and self._clear_of_obstacles(c) and self._clear_of_obstacles(e):
                return c, e
        return np.array([-3.0, 0.0]), np.array([3.0, 0.0])

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def step(self, actions: dict) -> tuple:
        cfg = self.cfg

        # 행동 적용
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for agent in AGENTS:
            a = np.clip(np.asarray(actions[agent], dtype=np.float64), -1.0, 1.0)
            ctrl[self.ctrl_slice[agent]] = a
        self.data.ctrl[:] = ctrl

        for _ in range(cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        cs = self._ant_state("chaser")
        es = self._ant_state("evader")
        c_xy, e_xy = cs['pos'][:2], es['pos'][:2]
        dist_ce = float(np.linalg.norm(e_xy - c_xy))

        # 배터리 갱신(도망자)
        _, e_charge_dist, _ = self._nearest_charger(e_xy)
        batt = self.battery.step(es['speed'], e_charge_dist)

        # 종료 판정
        caught = dist_ce < cfg.catch_radius
        depleted = batt['depleted']
        c_healthy = cfg.healthy_z_range[0] <= cs['pos'][2] <= cfg.healthy_z_range[1]
        e_healthy = cfg.healthy_z_range[0] <= es['pos'][2] <= cfg.healthy_z_range[1]
        truncated = self.step_count >= cfg.max_steps

        terminated = caught or depleted or (not c_healthy) or (not e_healthy)

        # 보상 계산
        rewards = self._compute_rewards(actions, cs, es, dist_ce, batt, caught, depleted, c_healthy, e_healthy)
        # progress shaping 기준점 갱신(보상 계산 직후)
        self._prev_dist_ce = dist_ce

        obs = self._get_obs()
        terminations = {a: terminated for a in AGENTS}
        truncations = {a: truncated for a in AGENTS}

        infos = {
            'chaser': {'distance': dist_ce, 'caught': caught},
            'evader': {
                'distance': dist_ce, 'battery': self.battery.normalized,
                'charging': batt['charging'], 'depleted': depleted,
                'caught': caught,
            },
        }
        return obs, rewards, terminations, truncations, infos

    def _compute_rewards(self, actions, cs, es, dist_ce, batt, caught, depleted, c_healthy, e_healthy) -> dict:
        cfg = self.cfg
        c_xy, e_xy = cs['pos'][:2], es['pos'][:2]

        # 도망자가 향하는(가장 가까운) 충전소
        e_charge_idx, e_charge_dist, e_charger = self._nearest_charger(e_xy)
        # 추격자가 그 충전소를 선점했는가
        chaser_charger_dist = float(np.linalg.norm(c_xy - e_charger))

        c_energy = float(np.sum(np.square(actions['chaser'])))
        e_energy = float(np.sum(np.square(actions['evader'])))

        # progress shaping: (이전거리 - 현재거리). +면 추격자가 접근, -면 도망자가 이탈.
        # 거리 변화량 기반(potential-based)이라 왔다갔다로 farming 불가 — telescoping 으로 누적합은
        # (초기거리 - 최종거리)에만 의존한다. 보행을 직접 유도하는 핵심 신호.
        prev_dist = getattr(self, "_prev_dist_ce", dist_ce)
        progress = prev_dist - dist_ce

        # 전방 속도(부호 있음): 제자리 진동은 상쇄되어 farming 불가
        c_fwd = float(cs['linvel'][:2] @ planar_forward_dir(cs['quat']))
        e_fwd = float(es['linvel'][:2] @ planar_forward_dir(es['quat']))

        # --- 추격자 보상 ---
        chaser_r = (
            cfg.chaser_progress_coef * progress
            + cfg.chaser_move_coef * c_fwd            # 보행 부트스트랩(전방 속도)
            - cfg.chaser_distance_coef * dist_ce
            - cfg.chaser_energy_coef * c_energy
            - cfg.chaser_time_penalty
        )
        # 블로킹 보상: 도망자도 그 충전소 근처(실제 선점 상황)일 때만 소량 부여 → 빈 충전소 캠핑 farming 차단
        if chaser_charger_dist < cfg.chaser_block_range and e_charge_dist < cfg.chaser_block_range:
            chaser_r += cfg.chaser_block_bonus
        if caught:
            chaser_r += cfg.chaser_catch_bonus
        if not c_healthy:
            chaser_r -= cfg.chaser_catch_bonus  # 넘어지면 큰 패널티

        # --- 도망자 보상 ---
        evader_r = (
            cfg.evader_progress_coef * (-progress)   # 멀어질수록(=progress<0) +
            + cfg.evader_move_coef * e_fwd           # 보행 부트스트랩(전방 속도)
            + cfg.evader_distance_coef * dist_ce
            + cfg.evader_battery_coef * self.battery.normalized
            - cfg.evader_energy_coef * e_energy
        )
        if batt['charging']:
            evader_r += cfg.evader_charge_bonus
        if caught:
            evader_r -= cfg.evader_caught_penalty
        if depleted:
            evader_r -= cfg.evader_empty_penalty
        if not e_healthy:
            evader_r -= cfg.evader_caught_penalty  # 넘어지면 큰 패널티

        return {'chaser': float(chaser_r), 'evader': float(evader_r)}

    # ------------------------------------------------------------------ #
    # 렌더링
    # ------------------------------------------------------------------ #
    def render(self, camera: str = "topdown", height: int = 720, width: int = 720):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
