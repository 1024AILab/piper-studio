import json
import time

import numpy as np
import pytest

from conftest import wait_until
from piper_studio.devices import ArmSample


IDS = ("0x155", "0x156", "0x157", "0x159")


@pytest.fixture
def pendant(engine, monkeypatch):
    """Fake passive CAN data with real acquisition/disk workers and simulated images."""
    q = np.asarray([0, 30000, -45000, 0, 25000, 0, 35000], dtype=np.float32)
    data = {"action": None, "follower": q.copy(), "ages": dict.fromkeys(IDS), "feedback_age": 0.0}
    engine.config.mode = "hardware_bus"

    def read():
        action = data["action"]
        ages = data["ages"].copy()
        complete = action is not None and all(age is not None for age in ages.values())
        return ArmSample(action.copy() if complete else None, data["follower"].copy(),
                         max(ages.values()) if complete else float("inf"),
                         data["feedback_age"], True, "", ages,
                         action.copy() if action is not None else None)

    monkeypatch.setattr(engine.robot, "read", read)
    wait_until(lambda: engine.sample.action is None)
    return engine, data, q


def fresh(data, q):
    data["action"] = q.copy()
    data["ages"] = dict.fromkeys(IDS, 0.01)
    data["follower"] = q.copy()


def test_click_before_motion_waits_then_records_and_saves_task_without_empty_frames(pendant):
    e, data, q = pendant
    assert e.snapshot()["can_start_record"] and not e.snapshot()["capture_ready"]
    assert "已准备录制" in e.start_record("抓起小黄鸭，放入笔筒")
    time.sleep(0.65)  # Longer than the previous target timeout.
    assert e.phase == "armed" and e.frames == 0 and not e.error
    fresh(data, q)
    wait_until(lambda: e.frames >= 4)
    assert e.phase == "recording"
    assert e.robot.send_count == 0
    e.save_record()
    signals = e.store.signals(0)
    assert signals["timestamp"][0] == 0
    np.testing.assert_array_equal(signals["actions"][0], q)
    np.testing.assert_array_equal(signals["state"][0], q)
    assert e.store.episodes()[0]["tasks"] == ["抓起小黄鸭，放入笔筒"]
    tasks = json.loads((e.store.root / "meta/tasks.jsonl").read_text())
    assert tasks["task"] == "抓起小黄鸭，放入笔筒"
    assert e.store.check(True)["valid"]


def test_partial_or_old_initial_targets_do_not_start_and_cancel_never_autostarts(pendant):
    e, data, q = pendant
    e.start_record("等待完整数据")
    fresh(data, q)
    data["ages"]["0x159"] = None
    time.sleep(0.15)
    assert e.phase == "armed" and e.frames == 0
    assert "夹爪" in e.snapshot()["target_status"]["message"]
    data["ages"]["0x159"] = 10
    time.sleep(0.15)
    assert e.phase == "armed" and e.frames == 0
    with pytest.raises(RuntimeError, match="尚未收到"):
        e.save_record()
    with pytest.raises(RuntimeError, match="录制或运动期间"):
        e.prepare(True)
    e.discard_record()
    fresh(data, q)
    time.sleep(0.2)
    assert e.phase == "idle" and e.frames == 0 and e.draft is None
    assert e.robot.send_count == 0


def test_stationary_hold_preserves_actual_target_and_true_age_then_accepts_new_targets(pendant):
    e, data, q = pendant
    fresh(data, q)
    wait_until(lambda: e.sample.action is not None)
    e.start_record("静止等待与继续示教")
    wait_until(lambda: e.frames >= 4)
    data["ages"] = dict.fromkeys(IDS, 5.0)
    before = e.frames
    wait_until(lambda: e.frames >= before + 4)
    assert e.phase == "recording"
    assert e.snapshot()["target_status"]["held_channels"] == list(range(7))
    new = q + [1500, 0, 0, 0, 0, 0, 0]
    fresh(data, new)
    wait_until(lambda: np.array_equal(e.previous_action, new))
    assert e.snapshot()["target_status"]["held_channels"] == []
    e.save_record()
    rows = [json.loads(line) for line in (e.store.support / "committed/episode_000000_frames.jsonl").read_text().splitlines()]
    held = [r for r in rows if r["held_target_channels"]]
    assert len(held) >= 3
    assert all(r["action_age_s"] == 5.0 and r["actions"] == q.tolist() for r in held)
    assert e.store.check(True)["valid"]
    assert e.robot.send_count == 0


def test_only_static_gripper_can_be_held_while_joint_commands_change(pendant):
    e, data, q = pendant
    fresh(data, q)
    wait_until(lambda: e.sample.action is not None)
    e.start_record("夹爪不变，关节继续示教")
    wait_until(lambda: e.frames >= 4)
    fresh(data, q + [2500, 0, 0, 0, 0, 0, 0])
    data["ages"]["0x159"] = 2.0
    before = e.frames
    wait_until(lambda: e.frames >= before + 3)
    assert e.phase == "recording"
    assert e.snapshot()["target_status"]["held_channels"] == [6]


@pytest.mark.parametrize("fault", ["stale_feedback", "wrong_position", "moving", "hold_disabled", "software_mode"])
def test_hold_never_masks_missing_feedback_unreached_targets_or_motion(pendant, fault):
    e, data, q = pendant
    fresh(data, q)
    wait_until(lambda: e.sample.action is not None)
    e.start_record("异常仍停止")
    wait_until(lambda: e.frames >= 4)
    with e.lock:
        data["ages"] = dict.fromkeys(IDS, 5.0)
        if fault == "stale_feedback":
            data["feedback_age"] = 5
        elif fault == "wrong_position":
            data["follower"] = q + [2000, 0, 0, 0, 0, 0, 0]
        elif fault == "moving":
            data["follower"] = q + [500, 0, 0, 0, 0, 0, 0]  # Within target tolerance, but moving.
        elif fault == "hold_disabled":
            e.config.hold_stationary_targets = False
        else:
            e.config.mode = "software"
    wait_until(lambda: e.phase == "draft")
    frames = e.frames
    time.sleep(0.15)
    assert e.frames == frames and e.draft is not None and e.error
    assert e.robot.send_count == 0


def test_stop_while_armed_and_camera_loss_do_not_autoresume(pendant):
    e, data, q = pendant
    e.start_record("等待中停止")
    e.stop()
    fresh(data, q)
    time.sleep(0.15)
    assert e.phase == "draft" and e.frames == 0
    e.discard_record()
    data["action"] = None
    data["ages"] = dict.fromkeys(IDS)
    wait_until(lambda: e.sample.action is None)
    e.start_record("等待中相机断开")
    e.cameras["wrist_image"].error = "test disconnect"
    wait_until(lambda: e.phase == "draft")
    assert "相机未就绪" in e.error
    assert e.frames == 0
