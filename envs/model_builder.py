"""
두 마리의 Ant 로봇 개 + 아레나 + 충전소로 구성된 MuJoCo 모델(XML) 생성

Gymnasium 의 단일 Ant-v4 모델을 기반으로, 본체/관절/액추에이터 이름에
접두사(`chaser_`, `evader_`)를 붙여 두 에이전트를 한 씬에 배치

비주얼:
- 본체는 광택 있는 구체 + 둥근 발(발 끝 구체)로 단순·깔끔하게 표현 (박테리오파지처럼 "머리(구체) + 다리섬유" 실루엣)
- 그림자·안티앨리어싱·다중 조명·반사 바닥·발광 충전소
- 발 끝 구체 등 장식 geom 은 `density=0, contype=0, conaffinity=0` 으로 두어
물리(질량/관성/충돌)와 관측 차원에 전혀 영향을 주지 않는다(순수 시각용)

- 두 Ant 의 다리 geom 은 서로 충돌하지 않도록 설정(conaffinity=0)
    - "포획" 판정은 물리 충돌이 아니라 거리 임계값으로 코드에서 처리
- 아레나 경계벽은 충돌체로 두어 에이전트가 맵 밖으로 나가지 못하게 함
"""
from typing import Sequence

# 순수 시각용(질량/충돌 없음) geom 공통 속성
_DECO = 'contype="0" conaffinity="0" density="0" group="0"'


def _leg(prefix: str, name: str, dx: float, dy: float, leg_mat: str, foot_mat: str) -> str:
    """
    아래로 향하는 강아지/로봇 다리 하나를 생성

    구조: 어깨(hip, z축 요잉) → 무릎(ankle, 다리 평면 굽힘) → 곧장 아래로
    내려가는 발. 옆으로 벌어지지 않고 몸통 아래로 내려가 곤충형 실루엣을
    피한다. 발끝(toe)은 포인트 색 + 실제 접지 패드(물리).
    """
    # 어깨까지 살짝 바깥으로(짧게)
    sx, sy     = 0.20 * dx, 0.20 * dy
    # 무릎(아래 다리 시작): 약간 바깥 + 아래
    kx, ky, kz = 0.10 * dx, 0.10 * dy, -0.10
    # 발끝: 거의 곧장 아래
    fx, fy, fz = 0.06 * dx, 0.06 * dy, -0.30
    return f"""
      <body name="{prefix}{name}_leg" pos="0 0 0">
        <geom name="{prefix}thigh_{name}" type="capsule" fromto="0 0 0 {sx} {sy} 0" size="0.075" material="{leg_mat}"/>
        <body name="{prefix}aux_{name}" pos="{sx} {sy} 0">
          <joint name="{prefix}hip_{name}" type="hinge" axis="0 0 1" pos="0 0 0" range="-40 40"/>
          <geom name="{prefix}shin_{name}" type="capsule" fromto="0 0 0 {kx} {ky} {kz}" size="0.075" material="{leg_mat}"/>
          <body pos="{kx} {ky} {kz}">
            <joint name="{prefix}ankle_{name}" type="hinge" axis="{-dy} {dx} 0" pos="0 0 0" range="-55 55"/>
            <geom name="{prefix}foot_{name}" type="capsule" fromto="0 0 0 {fx} {fy} {fz}" size="0.07" material="{leg_mat}"/>
            <geom name="{prefix}toe_{name}" type="sphere" pos="{fx} {fy} {fz}" size="0.085" material="{foot_mat}"/>
          </body>
        </body>
      </body>"""


def _mascot_body(prefix: str, body_mat: str, head_mat: str) -> str:
    """
    torso 를 통째로 감싸는 큰 큐브 마스코트 본체(시각용)

    큰 큐브가 다리 뿌리를 덮어 곤충/거미형 실루엣을 완화하고, 다리는
    큐브 아래로만 짧게 드러난다(귀여운 발). 큐브 정면(-y)에 얼굴을 둔다.
    구성: 큐브 몸통 + 검은 눈 2개 + 입 + 주황 밴드 + 밝은 뚜껑(lid)
    """
    return f"""
      <geom name="{prefix}body" type="box" pos="0 0 0.05" size="0.32 0.32 0.30" material="{body_mat}" {_DECO}/>
      <geom name="{prefix}band" type="box" pos="0 0 0.25" size="0.34 0.34 0.05" material="band_mat" {_DECO}/>
      <geom name="{prefix}lid" type="box" pos="0 0 0.345" size="0.325 0.325 0.045" material="{head_mat}" {_DECO}/>
      <geom name="{prefix}eye_l" type="sphere" pos="-0.13 -0.32 0.07" size="0.055" material="eye_mat" {_DECO}/>
      <geom name="{prefix}eye_r" type="sphere" pos="0.13 -0.32 0.07" size="0.055" material="eye_mat" {_DECO}/>
      <geom name="{prefix}pupil_l" type="sphere" pos="-0.13 -0.36 0.075" size="0.022" material="eyeshine_mat" {_DECO}/>
      <geom name="{prefix}pupil_r" type="sphere" pos="0.13 -0.36 0.075" size="0.022" material="eyeshine_mat" {_DECO}/>
      <geom name="{prefix}mouth" type="box" pos="0 -0.321 -0.08" size="0.075 0.01 0.018" material="eye_mat" {_DECO}/>"""


def _ant_body(prefix: str, pos: Sequence[float], body_mat: str, leg_mat: str, head_mat: str) -> str:
    """
    접두사가 붙은 본체(물리 torso + 큐브 마스코트 외형 + 4 legs)를 생성

    물리는 중심의 구형 torso(질량/충돌)가 담당하고, 큰 큐브 외형이 이를
    감싸 귀여운 마스코트로 보이게 한다(외형은 모두 비충돌·무질량).
    """
    x, y, z = pos
    legs = (
        _leg(prefix, "1", 1, 1, leg_mat, "foot_mat")
        + _leg(prefix, "2", -1, 1, leg_mat, "foot_mat")
        + _leg(prefix, "3", -1, -1, leg_mat, "foot_mat")
        + _leg(prefix, "4", 1, -1, leg_mat, "foot_mat")
    )
    return f"""
    <body name="{prefix}torso" pos="{x} {y} {z}">
      <camera name="{prefix}track" mode="trackcom" pos="0 -4 2.2" xyaxes="1 0 0 0 1 1.4"/>
      <geom name="{prefix}torso_geom" pos="0 0 0" size="0.25" type="sphere" material="{body_mat}"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="{prefix}root" pos="0 0 0" type="free"/>
{_mascot_body(prefix, body_mat, head_mat)}
      {legs}
    </body>"""


def _actuators(prefix: str) -> str:
    """
    한 Ant 의 8개 모터 액추에이터
    """
    joints = ["hip_4", "ankle_4", "hip_1", "ankle_1",
              "hip_2", "ankle_2", "hip_3", "ankle_3"]
    return "\n".join(
        f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" '
        f'joint="{prefix}{j}" gear="150"/>'
        for j in joints
    )


def _walls(arena_size: float) -> str:
    """
    아레나 경계벽(충돌체) 4개 — 저폴리 스톤 + 상단 캡
    """
    s, h, t = arena_size, 0.7, 0.25
    # (이름, x, y, half_x, half_y)
    walls = [
        ("wall_n", 0.0, s, s + t, t),
        ("wall_s", 0.0, -s, s + t, t),
        ("wall_e", s, 0.0, t, s + t),
        ("wall_w", -s, 0.0, t, s + t),
    ]
    out = []
    for n, x, y, hx, hy in walls:
        out.append(
            f'    <geom name="{n}" type="box" pos="{x} {y} {h}" '
            f'size="{hx} {hy} {h}" contype="1" conaffinity="1" '
            f'material="wall_mat"/>')
        out.append(
            f'    <geom name="{n}_cap" type="box" pos="{x} {y} {2 * h}" '
            f'size="{hx} {hy} 0.07" material="wall_cap_mat" {_DECO}/>')
    return "\n".join(out)


def _pillar(idx: int, x: float, y: float, h: float = 1.3) -> str:
    """
    저폴리 스톤 기둥(받침 + 기둥 + 머리) / 배경 장식(비충돌)
    """
    half = h / 2
    return (
        f'    <geom name="pillar_base_{idx}" type="box" pos="{x} {y} 0.1" '
        f'size="0.55 0.55 0.1" material="stone_mat" {_DECO}/>\n'
        f'    <geom name="pillar_shaft_{idx}" type="cylinder" '
        f'pos="{x} {y} {half + 0.2}" size="0.36 {half}" material="stone_mat" {_DECO}/>\n'
        f'    <geom name="pillar_cap_{idx}" type="box" '
        f'pos="{x} {y} {h + 0.3}" size="0.55 0.55 0.1" material="stone_mat" {_DECO}/>'
    )


def _scenery(arena_size: float) -> str:
    """
    벽 바깥 배경 장식: 기둥 / 보라 크리스탈 / 주황 포털 / 바위

    모두 정적·비충돌(`_DECO`)이라 에이전트 물리/학습에 영향 없음.
    'angle' 카메라가 바라보는 +y(뒤쪽)에 주로 배치한다.
    """
    s = arena_size
    parts = []

    # 뒤쪽/측면 기둥들
    for i, (px, py) in enumerate([
        (-7.0, s + 2.5), (-1.5, s + 3.2), (4.5, s + 2.6),
        (-(s + 2.5), 2.5), (s + 2.5, -3.0),
    ]):
        parts.append(_pillar(i, px, py, h=1.4 + 0.2 * (i % 3)))

    # 보라 크리스탈 군집(오른쪽 뒤)
    crystals = [
        (s + 2.2, s - 1.0, 1.7, "12 0 18"),
        (s + 3.0, s - 0.2, 1.2, "-14 0 -22"),
        (s + 1.4, s + 0.4, 2.1, "6 0 8"),
        (s + 3.4, s - 1.8, 1.3, "16 0 35"),
        (s + 2.0, -7.0, 1.5, "10 0 -12"),
        (s + 2.8, -7.6, 1.0, "-8 0 24"),
    ]
    for i, (cx, cy, ch, eu) in enumerate(crystals):
        parts.append(
            f'    <geom name="crystal_{i}" type="box" pos="{cx} {cy} {ch * 0.5}" '
            f'size="0.26 0.26 {ch}" euler="{eu}" material="crystal_mat" {_DECO}/>')

    # 주황 포털(왼쪽 뒤): 스톤 프레임 + 발광 패널
    pgx, pgy = -(s - 1.0), s + 1.6
    parts.append(
        f'    <geom name="portal_panel" type="box" pos="{pgx} {pgy} 1.05" '
        f'size="0.9 0.12 1.0" material="portal_mat" {_DECO}/>')
    parts.append(
        f'    <geom name="portal_l" type="box" pos="{pgx - 1.05} {pgy} 1.1" '
        f'size="0.18 0.3 1.25" material="stone_mat" {_DECO}/>')
    parts.append(
        f'    <geom name="portal_r" type="box" pos="{pgx + 1.05} {pgy} 1.1" '
        f'size="0.18 0.3 1.25" material="stone_mat" {_DECO}/>')
    parts.append(
        f'    <geom name="portal_top" type="box" pos="{pgx} {pgy} 2.45" '
        f'size="1.3 0.3 0.22" material="stone_mat" {_DECO}/>')

    # 바위(저폴리 보울더)
    for i, (bx, by, br) in enumerate([
        (-(s + 2.0), -6.0, 0.8), (s + 1.5, s + 1.5, 0.7), (-4.0, s + 4.0, 0.6),
    ]):
        parts.append(
            f'    <geom name="boulder_{i}" type="ellipsoid" '
            f'pos="{bx} {by} {br * 0.6}" size="{br} {br * 0.85} {br * 0.7}" '
            f'euler="0 0 {20 * i}" material="boulder_mat" {_DECO}/>')

    return "\n".join(parts)


def _chargers(charger_positions: Sequence[Sequence[float]], charge_range: float) -> str:
    """
    충전소(에너지 사당): 스톤 단 + 발광 림 + 녹색 에너지 크리스탈 + 글로우 링

    배경 장식과 같은 저폴리 컨셉. 모두 비충돌(시각용). 충전 판정은 코드에서
    충전소 (x, y) 거리로 처리한다.
    """
    out = []
    gems = [(0.0, 0.0, 1.0, "0 0 0"),
            (0.20, 0.10, 0.7, "16 0 22"),
            (-0.18, 0.14, 0.6, "-14 0 -26")]
    for i, (cx, cy) in enumerate(charger_positions):
        # 스톤 받침 + 발광 림
        out.append(
            f'    <geom name="charger_base_{i}" type="cylinder" '
            f'pos="{cx} {cy} 0.08" size="0.85 0.08" material="stone_mat" {_DECO}/>')
        out.append(
            f'    <geom name="charger_rim_{i}" type="cylinder" '
            f'pos="{cx} {cy} 0.17" size="0.7 0.04" material="charger_mat" {_DECO}/>')
        # 중앙 녹색 에너지 크리스탈 군집
        for j, (dx, dy, gh, eu) in enumerate(gems):
            out.append(
                f'    <geom name="charger_gem_{i}_{j}" type="box" '
                f'pos="{cx + dx} {cy + dy} {0.2 + gh * 0.5}" '
                f'size="0.13 0.13 {gh}" euler="{eu}" '
                f'material="charger_mat" {_DECO}/>')
        # 바닥 글로우 링(충전 범위 표시)
        out.append(
            f'    <geom name="charger_range_{i}" type="cylinder" '
            f'pos="{cx} {cy} 0.02" size="{charge_range} 0.012" '
            f'material="range_mat" {_DECO}/>')
    return "\n".join(out)


def _obstacles(positions: Sequence[Sequence[float]], radius: float, height: float) -> str:
    """
    아레나 내부 실제 장애물 기둥(충돌)
    샤프트는 충돌체, 받침/머리/보석은 장식

    저폴리 스톤 기둥 + 상단 보라 크리스탈. 에이전트가 부딪히고 뒤에 숨을 수 있다.
    """
    out = []
    half = height / 2.0
    cap = radius + 0.22
    for i, (x, y) in enumerate(positions):
        out.append(
            f'    <geom name="obstacle_base_{i}" type="box" pos="{x} {y} 0.12" '
            f'size="{cap} {cap} 0.12" material="stone_mat" {_DECO}/>')
        out.append(
            f'    <geom name="obstacle_{i}" type="cylinder" '
            f'pos="{x} {y} {half}" size="{radius} {half}" '
            f'contype="1" conaffinity="1" material="stone_mat"/>')
        out.append(
            f'    <geom name="obstacle_cap_{i}" type="box" '
            f'pos="{x} {y} {height + 0.1}" size="{cap} {cap} 0.1" '
            f'material="stone_mat" {_DECO}/>')
        out.append(
            f'    <geom name="obstacle_gem_{i}" type="box" '
            f'pos="{x} {y} {height + 0.45}" size="0.2 0.2 0.32" euler="0 0 45" '
            f'material="crystal_mat" {_DECO}/>')
    return "\n".join(out)


def build_model_xml(
    chaser_pos: Sequence[float] = (-3.0, 0.0, 0.75),
    evader_pos: Sequence[float] = (3.0, 0.0, 0.75),
    charger_positions: Sequence[Sequence[float]] = ((6.0, 6.0), (-6.0, -6.0)),
    arena_size: float = 10.0,
    charge_range: float = 1.5,
    obstacle_positions: Sequence[Sequence[float]] = (),
    obstacle_radius: float = 0.5,
    obstacle_height: float = 2.0,
) -> str:
    """
    전체 MuJoCo 모델 XML 문자열을 생성

    Args:
        chaser_pos:        추격자 초기 위치 (x, y, z)
        evader_pos:        도망자 초기 위치 (x, y, z)
        charger_positions: 충전소들의 (x, y) 좌표 리스트
        arena_size:        아레나 반경(중심에서 벽까지) / 평면은 그 두 배
        charge_range:      충전 가능 거리(시각 표시에 사용)

    Returns:
        MuJoCo 가 `from_xml_string` 으로 로드 가능한 XML 문자열.
    """
    floor_size = arena_size + 6.0
    chaser = _ant_body("chaser_", chaser_pos, body_mat="chaser_mat",
                       leg_mat="chaser_leg_mat", head_mat="chaser_head_mat")
    evader = _ant_body("evader_", evader_pos, body_mat="evader_mat",
                       leg_mat="evader_leg_mat", head_mat="evader_head_mat")

    return f"""<mujoco model="ant_chase">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>

  <visual>
    <headlight ambient="0.30 0.26 0.36" diffuse="0.28 0.26 0.34" specular="0.1 0.1 0.12"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map force="0.1" znear="0.02" zfar="60" haze="0.12"/>
    <global offwidth="1280" offheight="720" azimuth="120" elevation="-22"/>
    <rgba haze="0.44 0.36 0.56 1"/>
  </visual>

  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5"
          margin="0.01"/>
  </default>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.50 0.40 0.66" rgb2="0.16 0.12 0.26" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.20 0.20 0.30" rgb2="0.15 0.15 0.24"
             width="512" height="512" mark="edge" markrgb="0.45 0.42 0.62"/>
    <material name="floor_mat" texture="grid" texrepeat="22 22"
              texuniform="true" reflectance="0.45" shininess="0.7" specular="0.7"/>
    <material name="wall_mat" rgba="0.46 0.43 0.58 1" specular="0.4"
              shininess="0.3" reflectance="0.05"/>
    <material name="wall_cap_mat" rgba="0.62 0.58 0.74 1" specular="0.5"
              shininess="0.5"/>
    <material name="stone_mat" rgba="0.66 0.63 0.76 1" specular="0.3"
              shininess="0.3" reflectance="0.05"/>
    <material name="boulder_mat" rgba="0.40 0.37 0.50 1" specular="0.2"
              shininess="0.2"/>
    <material name="crystal_mat" rgba="0.72 0.32 0.95 1" emission="0.55"
              specular="0.9" shininess="0.9"/>
    <material name="portal_mat" rgba="1.0 0.60 0.18 1" emission="0.85"
              specular="0.5" shininess="0.5"/>

    <material name="chaser_mat" rgba="0.86 0.22 0.24 1" specular="0.7"
              shininess="0.7" reflectance="0.08"/>
    <material name="chaser_head_mat" rgba="0.95 0.40 0.32 1" specular="0.8"
              shininess="0.8" reflectance="0.1"/>
    <material name="chaser_leg_mat" rgba="0.62 0.14 0.16 1" specular="0.5"
              shininess="0.5"/>
    <material name="evader_mat" rgba="0.24 0.46 0.92 1" specular="0.7"
              shininess="0.7" reflectance="0.08"/>
    <material name="evader_head_mat" rgba="0.40 0.66 1.0 1" specular="0.8"
              shininess="0.8" reflectance="0.1"/>
    <material name="evader_leg_mat" rgba="0.16 0.30 0.66 1" specular="0.5"
              shininess="0.5"/>

    <material name="eye_mat" rgba="0.07 0.07 0.09 1" specular="0.3"
              shininess="0.4"/>
    <material name="eyeshine_mat" rgba="0.98 0.98 1.0 1" specular="0.9"
              shininess="1.0" emission="0.25"/>
    <material name="band_mat" rgba="0.98 0.62 0.16 1" specular="0.5"
              shininess="0.5"/>
    <material name="foot_mat" rgba="1.0 0.85 0.25 1" specular="0.6"
              shininess="0.6"/>
    <material name="charger_mat" rgba="0.26 0.58 1.0 1" emission="0.6"
              specular="0.7" shininess="0.7"/>
    <material name="range_mat" rgba="0.45 0.50 0.98 0.18" emission="0.45"/>
  </asset>

  <worldbody>
    <light name="sun" directional="true" castshadow="true"
           pos="4 -6 11" dir="-0.35 0.5 -1"
           diffuse="0.58 0.55 0.62" specular="0.3 0.3 0.35"/>
    <light name="warm" directional="false" castshadow="false"
           pos="-6 5 5.5" dir="0.35 -0.35 -1" cutoff="80" exponent="3"
           diffuse="1.3 0.7 0.25" specular="0.6 0.35 0.12"/>
    <light name="cool" directional="false" castshadow="false"
           pos="{arena_size - 1} 5 5.5" dir="-0.35 -0.3 -1" cutoff="80" exponent="3"
           diffuse="0.95 0.30 1.2" specular="0.5 0.18 0.6"/>

    <geom name="floor" type="plane" material="floor_mat" pos="0 0 0"
          size="{floor_size} {floor_size} 0.5" condim="3"
          contype="1" conaffinity="1"/>
    <camera name="topdown" mode="fixed" pos="0 0 22" xyaxes="1 0 0 0 1 0"/>
    <camera name="angle" mode="fixed" pos="0 -16 14" xyaxes="1 0 0 0 0.66 0.75"/>
{_walls(arena_size)}
{_scenery(arena_size)}
{_obstacles(obstacle_positions, obstacle_radius, obstacle_height)}
{_chargers(charger_positions, charge_range)}
{chaser}
{evader}
  </worldbody>

  <actuator>
{_actuators("chaser_")}
{_actuators("evader_")}
  </actuator>
</mujoco>"""
