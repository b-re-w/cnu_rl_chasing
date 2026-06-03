"""BatterySystem 단위 테스트"""

from envs.battery_system import BatterySystem


def test_drains_with_speed():
    b = BatterySystem(drain_rate=0.1, idle_drain=0.0, charge_range=1.5)
    b.reset()
    out = b.step(speed=2.0, distance_to_charger=10.0)  # 충전소 밖
    assert out['battery'] < 1.0
    assert not out['charging']
    assert abs(out['battery'] - (1.0 - 2.0 * 0.1)) < 1e-9


def test_idle_drain():
    b = BatterySystem(drain_rate=0.0, idle_drain=0.01)
    b.reset()
    out = b.step(speed=0.0, distance_to_charger=10.0)
    assert out['battery'] < 1.0  # 정지여도 자연 방전


def test_charges_within_range():
    b = BatterySystem(drain_rate=0.0, idle_drain=0.0, charge_rate=0.05, charge_range=1.5)
    b.reset(battery=0.5)
    out = b.step(speed=0.0, distance_to_charger=1.0)  # 충전소 안
    assert out['charging']
    assert out['battery'] > 0.5


def test_charge_caps_at_max():
    b = BatterySystem(charge_rate=0.5, charge_range=1.5)
    b.reset(battery=0.9)
    out = b.step(speed=0.0, distance_to_charger=0.5)
    assert out['battery'] <= b.max_battery


def test_depletes_to_zero():
    b = BatterySystem(drain_rate=1.0, idle_drain=0.0)
    b.reset(battery=0.05)
    out = b.step(speed=1.0, distance_to_charger=10.0)
    assert out['battery'] == 0.0
    assert out['depleted']


def test_normalized():
    b = BatterySystem(max_battery=2.0)
    b.reset(battery=1.0)
    assert abs(b.normalized - 0.5) < 1e-9
