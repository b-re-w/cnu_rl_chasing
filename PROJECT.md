# 🐕 강화학습 기반 로봇 개 추격-도망 시뮬레이션

## 프로젝트 개요

MuJoCo 물리 엔진 기반의 커스텀 멀티에이전트 환경에서,  
**추격자(Chaser)** 와 **도망자(Evader)** 로봇 개 두 마리가 서로 경쟁하며 강화학습을 통해 전략적 행동을 학습한다.

도망자는 **배터리 시스템**을 갖추고 있어 이동할수록 에너지가 소모되며,  
맵에 배치된 **충전소**에 도달해 배터리를 충전해야 생존할 수 있다.  
추격자는 도망자를 잡거나 충전소 접근을 방해하는 전략을 학습하고,  
두 에이전트는 **공진화(Co-evolution)** 를 통해 점점 정교한 행동을 창발한다.

---

## 핵심 아이디어 및 독창성

### 배터리 연동 생존 메커니즘
- 도망자의 배터리는 이동 속도에 비례하여 소모된다
- 배터리가 완전히 방전되면 에피소드가 종료된다
- 도망자는 단순히 도망치는 것이 아니라 **에너지 관리 + 회피** 를 동시에 학습해야 한다

### 충전소를 둘러싼 전략적 행동
- 맵에 충전소가 배치되어 있으며 도망자는 충전소를 향해 이동하려 한다
- 추격자는 도망자를 잡는 것뿐만 아니라 **충전소 진입을 차단**하는 전략을 학습한다
- 단순한 추격-도망을 넘어 **영역 제어, 경로 예측, 기만 행동** 등이 자연스럽게 창발된다

### 멀티에이전트 공진화
- 추격자가 강해지면 도망자도 더 영리한 도망 경로를 학습한다
- 도망자가 교묘해지면 추격자도 더 정교한 차단 전략을 학습한다
- 두 에이전트의 상호 경쟁이 학습의 핵심 동력이다

---

## 환경 구성

### 시뮬레이터 및 라이브러리
| 항목 | 선택 |
|------|------|
| 물리 엔진 | MuJoCo |
| Gym 인터페이스 | Gymnasium |
| 멀티에이전트 래핑 | PettingZoo |
| 기반 로봇 모델 | Ant-v4 커스터마이징 |

### 커스텀 환경 구성 요소

```
AntChaseEnv (Custom Gymnasium Environment)
├── 에이전트
│   ├── Chaser  : Ant-v4 기반, 도망자 추격 및 충전소 차단
│   └── Evader  : Ant-v4 기반, 배터리 시스템 탑재, 충전소 도달 목표
├── 맵
│   ├── 평면 지형 (기본)
│   └── 장애물 배치 (선택)
└── 충전소
    ├── 맵에 고정 or 랜덤 배치
    └── 근접 시 배터리 충전
```

### Observation Space

```python
# 추격자 관측
chaser_obs = [
    자신의 위치(x, y, z),
    자신의 속도(vx, vy, vz),
    관절 각도 및 각속도,
    도망자까지의 상대 위치 및 거리,
    충전소까지의 상대 위치 및 거리,
]

# 도망자 관측
evader_obs = [
    자신의 위치(x, y, z),
    자신의 속도(vx, vy, vz),
    관절 각도 및 각속도,
    추격자까지의 상대 위치 및 거리,
    충전소까지의 상대 위치 및 거리,
    현재 배터리 잔량,          # 핵심 추가 변수
    충전소 방향 벡터,           # 핵심 추가 변수
]
```

### Reward 설계

```python
# 추격자 보상
chaser_reward = (
    + 10.0  * catch_bonus              # 도망자 포획 시 큰 보상
    -  0.1  * distance_to_evader       # 거리가 가까울수록 보상
    +  3.0  * block_charger_bonus      # 충전소 근처 선점 보상
    -  0.01 * energy_consumption       # 에너지 효율 패널티
)

# 도망자 보상
evader_reward = (
    +  0.1  * distance_from_chaser     # 추격자와 거리 유지 보상
    +  5.0  * reach_charger_bonus      # 충전소 도달 시 보상
    +  0.1  * battery_efficiency       # 배터리 절약 보상
    - 10.0  * caught_penalty           # 잡혔을 때 큰 패널티
    -  5.0  * battery_empty_penalty    # 배터리 방전 시 패널티
)
```

### 배터리 시스템

```python
class BatterySystem:
    max_battery  = 1.0
    drain_rate   = velocity * 0.01      # 속도에 비례한 소모
    charge_rate  = 0.05                 # 충전소 근접 시 충전량 (스텝당)
    charge_range = 1.5                  # 충전 가능 거리 임계값

    def step(self, velocity, distance_to_charger):
        self.battery -= velocity * self.drain_rate
        if distance_to_charger < self.charge_range:
            self.battery = min(1.0, self.battery + self.charge_rate)
        if self.battery <= 0:
            return done=True
```

---

## 학습 알고리즘

세 가지 알고리즘 구성을 동일 환경에서 학습시켜 성능을 비교한다.

### PPO (Proximal Policy Optimization)
- On-policy 알고리즘
- 안정적인 학습, 하이퍼파라미터에 덜 민감
- 추격자/도망자 모두 PPO로 학습 (대칭 구성)

### SAC (Soft Actor-Critic)
- Off-policy 알고리즘, 최대 엔트로피 강화학습
- 연속 행동 공간에서 sample efficiency가 높음
- 추격자/도망자 모두 SAC로 학습 (대칭 구성)

### Asymmetric Self-play (제안 알고리즘) ⭐
추격자와 도망자를 **서로 다른 알고리즘**으로 학습시키는 비대칭 구성이다.

```
Chaser  : PPO  → 안정적이고 보수적인 추격 전략
Evader  : SAC  → 탐험적이고 유연한 회피 전략
```

**핵심 아이디어**
- 추격자와 도망자는 본질적으로 다른 목표를 가진다
- 추격자는 **수렴적(convergent)** 행동이 유리 → PPO의 안정성이 적합
- 도망자는 **다양한 탐험(exploratory)** 행동이 유리 → SAC의 최대 엔트로피가 적합
- 역할의 비대칭성을 알고리즘 수준에서도 반영하는 것이 더 자연스럽다는 가설

**연구 질문**
> "알고리즘의 비대칭 구성이 공진화 속도와 최종 성능에 영향을 미치는가?"

**구현 방식**
```python
# 역할별로 다른 알고리즘 할당
chaser_agent = PPOAgent(obs_dim, act_dim)
evader_agent = SACAgent(obs_dim, act_dim)

# Self-play: 주기적으로 상대 정책 스냅샷 갱신
if episode % snapshot_interval == 0:
    chaser_snapshot = copy(chaser_agent.policy)
    evader_snapshot = copy(evader_agent.policy)
```

### 비교 지표
| 지표 | 설명 |
|------|------|
| 포획률 | 에피소드당 추격자가 도망자를 잡는 비율 |
| 생존 시간 | 도망자가 배터리를 유지하며 생존하는 평균 시간 |
| 충전 성공률 | 도망자가 충전소에 도달하는 비율 |
| 학습 수렴 속도 | 목표 성능 도달까지 필요한 학습 스텝 수 |
| 보상 곡선 | 에피소드별 누적 보상 추이 |
| 공진화 속도 | 양측 에이전트 성능이 함께 향상되는 속도 |

---

## 기대 결과 및 창발 행동

- **추격자**: 충전소 앞을 선점하여 도망자의 충전을 막는 전략 학습
- **도망자**: 배터리가 낮을 때 충전소로 향하는 최단 경로 탐색, 추격자를 따돌리는 페인트 동작 학습
- **공진화**: 에피소드가 진행될수록 양측의 전략이 점점 정교해지는 군비경쟁(arms race) 현상 관찰

---

## 프로젝트 구조

```
project/
├── envs/
│   ├── ant_chase_env.py         # 커스텀 멀티에이전트 환경
│   ├── battery_system.py        # 배터리 및 충전소 로직
│   └── multiagent_wrapper.py    # PettingZoo 래핑
├── agents/
│   ├── ppo_agent.py             # PPO 에이전트
│   └── sac_agent.py             # SAC 에이전트
├── train.py                     # 학습 실행 스크립트
├── evaluate.py                  # 성능 평가 스크립트
├── visualize.py                 # 시각화 및 영상 저장
└── PROJECT.md                   # 프로젝트 설명서
```

---

## 참고 문헌

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning*, 2018
- Baker et al., *Emergent Tool Use From Multi-Agent Autocurricula (OpenAI Hide and Seek)*, 2019
- Terry et al., *PettingZoo: Gym for Multi-Agent Reinforcement Learning*, 2021
- Bansal et al., *Emergent Complexity via Multi-Agent Competition*, 2018
