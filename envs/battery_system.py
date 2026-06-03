"""
도망자(Evader)의 배터리 및 충전소 로직
- 배터리는 이동 속도에 비례하여 소모
- 충전소 근접(charge_range 이내) 시 스텝당 charge_rate 만큼 충전
- 배터리가 0 이하가 되면 방전(done)으로 처리
"""

from dataclasses import dataclass


@dataclass
class BatterySystem:
    """
    속도 기반 소모 + 충전소 근접 충전 모델

    Attributes:
        max_battery:  배터리 최대치(정규화 1.0)
        drain_rate:   속도 1단위당 스텝마다 소모되는 배터리량 계수
        idle_drain:   정지 상태에서도 소모되는 최소 배터리량(자연 방전)
        charge_rate:  충전소 근접 시 스텝당 충전량
        charge_range: 충전 가능 거리 임계값
        battery:      현재 배터리 잔량(0~max_battery)
    """

    max_battery: float = 1.0
    drain_rate: float = 0.01
    idle_drain: float = 0.0005
    charge_rate: float = 0.05
    charge_range: float = 1.5
    battery: float = 1.0

    def reset(self, battery: float | None = None) -> float:
        self.battery = self.max_battery if battery is None else float(battery)
        return self.battery

    def step(self, speed: float, distance_to_charger: float) -> dict:
        """
        한 스텝 진행하고 배터리 상태를 갱신

        Args:
            speed:               도망자 본체의 속력(스칼라, m/s)
            distance_to_charger: 가장 가까운 충전소까지의 거리

        Returns:
            dict: {
                "battery": float,        # 갱신된 잔량
                "charging": bool,        # 이번 스텝에 충전했는지
                "depleted": bool,        # 방전 여부(에피소드 종료 신호)
                "drained": float,        # 이번 스텝에 소모한 양
                "charged": float,        # 이번 스텝에 충전한 양
            }
        """
        prev = self.battery

        # 1) 속도 비례 소모 + 자연 방전
        drain = abs(speed) * self.drain_rate + self.idle_drain
        self.battery = max(0.0, self.battery - drain)

        # 2) 충전소 근접 시 충전
        charging = distance_to_charger < self.charge_range
        charged = 0.0
        if charging and self.battery > 0.0:
            new_battery = min(self.max_battery, self.battery + self.charge_rate)
            charged = new_battery - self.battery
            self.battery = new_battery

        depleted = self.battery <= 0.0
        return {
            'battery': self.battery,
            'charging': bool(charging),
            'depleted': bool(depleted),
            'drained': prev - self.battery + charged,  # 순소모 추적용
            'charged': charged,
        }

    @property
    def normalized(self) -> float:
        """
        0~1 로 정규화된 배터리 잔량 (관측값으로 사용)
        """
        return self.battery / self.max_battery
