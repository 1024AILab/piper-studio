import time

import pytest

from conftest import wait_until
from piper_studio.can_health import CanHealth, decode_can_event


def payload(controller=0, tx=0, rx=0):
    return bytes([0, controller, 0, 0, 0, 0, tx, rx])


def receive_for(health, start, duration=1.2):
    for index in range(int(duration / 0.02) + 1):
        health.feed_valid(now=start + index * 0.02)


def test_error_active_notification_is_normal_status_not_a_fault():
    health = CanHealth()
    health.feed_error(0x204, payload(0x40), now=10)
    snapshot = health.snapshot(now=10.1)
    assert snapshot["error"] == ""
    assert snapshot["error_frames"] == 0 and snapshot["status_frames"] == 1
    assert "Error-active" in snapshot["recent_events"][0]["message"]


def test_overflow_is_reported_with_raw_details_then_recovers_after_clean_traffic():
    health = CanHealth()
    health.feed_error(4, payload(1), now=10)
    snapshot = health.snapshot(now=10)
    assert "接收缓冲区溢出" in snapshot["error"]
    assert snapshot["recent_events"][0]["data"] == "00 01 00 00 00 00 00 00"
    receive_for(health, 10.05, 0.5)
    assert health.snapshot(now=10.55)["error"]  # A few good frames do not erase it.
    receive_for(health, 10.55, 0.7)
    assert health.snapshot(now=11.25)["error"] == ""
    assert health.snapshot(now=11.25)["state"] == "已恢复"


def test_elapsed_time_or_sporadic_rx_alone_never_clears_error():
    health = CanHealth()
    health.feed_error(4, payload(1), now=10)
    assert health.snapshot(now=50)["error"]
    for index in range(40):
        health.feed_valid(now=50 + index)
    assert health.snapshot(now=89)["error"]


@pytest.mark.parametrize("mask,data", [(0x40, payload()), (4, payload(0x10)), (4, payload(0x20)), (0x4000, payload())])
def test_critical_or_unknown_error_requires_explicit_active_notice(mask, data):
    health = CanHealth()
    health.feed_error(mask, data, now=10)
    receive_for(health, 10.1, 2)
    assert health.snapshot(now=12.1)["error"]
    health.feed_error(4, payload(0x40), now=12.2)
    health.feed_valid(now=12.2)
    assert health.snapshot(now=12.2)["error"] == ""


def test_repeated_faults_stay_blocked_and_keep_counters_without_log_flood():
    health = CanHealth()
    for index in range(100):
        health.feed_error(4, payload(1), now=10 + index * 0.001)
        health.feed_valid(now=10 + index * 0.001)
    snapshot = health.snapshot(now=10.1)
    assert snapshot["error"] and snapshot["error_frames"] == 100
    assert len(snapshot["recent_events"]) == 1 and snapshot["recent_events"][0]["repeats"] == 100


def test_error_active_bit_does_not_hide_concurrent_fault_bits():
    assert decode_can_event(4, payload(0x41))["blocking"]
    assert decode_can_event(0x44, payload(0x40))["needs_active"]
    assert decode_can_event(0x204, payload(0x40, rx=128))["blocking"]


def test_restart_waits_for_stream_stability_and_counts_as_status():
    health = CanHealth()
    health.feed_error(0x40, payload(), now=10)
    health.feed_error(0x100, payload(), now=10.1)
    assert "重启" in health.snapshot(now=10.1)["error"]
    receive_for(health, 10.2)
    snapshot = health.snapshot(now=11.4)
    assert snapshot["error"] == "" and snapshot["error_frames"] == 1 and snapshot["status_frames"] == 1


def test_missing_receive_is_visible_even_without_an_error_notification():
    health = CanHealth()
    health.feed_valid(now=10)
    assert health.snapshot(now=11)["state"] == "无新数据"


def test_real_bus_fault_stops_recording_but_recovery_never_resumes_it(engine, monkeypatch):
    health = CanHealth()
    read_original = engine.robot.read
    def read():
        sample = read_original()
        health.feed_valid()
        sample.error = health.snapshot()["error"]
        return sample
    monkeypatch.setattr(engine.robot, "read", read)
    engine.start_record("CAN error and recovery")
    wait_until(lambda: engine.frames >= 2)
    health.feed_error(4, payload(1))
    wait_until(lambda: engine.phase == "draft")
    frames = engine.frames
    # The health decoder recovers before the 50 Hz engine publishes its next
    # sample. Wait for both, rather than racing that cross-thread handoff.
    wait_until(lambda: not health.snapshot()["error"] and not engine.sample.error)
    assert engine.phase == "draft" and engine.frames == frames
    assert engine.robot.send_count == 0
    engine.save_record()
    engine.start_record("explicit next episode after recovery")
    wait_until(lambda: engine.frames >= 2)
    assert engine.phase == "recording"
