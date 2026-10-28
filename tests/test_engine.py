import time

import numpy as np
import pytest

from conftest import wait_until


def record_short(engine, n=8):
    engine.start_record("SIMULATION: pick and place")
    wait_until(lambda: engine.frames >= n or engine.phase != "recording")
    assert engine.phase == "recording", engine.error
    engine.save_record()


def test_capture_save_seek_and_visual_replay_never_send_commands(engine):
    record_short(engine)
    result = engine.store.check(True)
    assert result["valid"]
    assert result["previous_action_matches"] == result["frames"] - 1
    signal = engine.store.signals(0)
    np.testing.assert_array_equal(signal["state"][0], signal["actions"][0])
    assert engine.robot.send_count == 0
    n = engine.load_preview(0)
    engine.seek_preview(n - 1)
    assert engine.preview()["frame_index"] == n - 1
    engine.play_preview(1)
    time.sleep(0.15)
    assert engine.preview()["frame_index"] > 0
    assert engine.robot.send_count == 0


def test_camera_freeze_stops_recording_and_keeps_draft(engine):
    engine.start_record("camera failure test")
    wait_until(lambda: engine.frames >= 2)
    engine.cameras["wrist_image"].stop_event.set()
    engine.cameras["wrist_image"].thread.join(1)
    wait_until(lambda: engine.phase == "draft")
    assert "未产生新图像" in engine.error or "相机未就绪" in engine.error
    before = engine.frames
    time.sleep(0.2)
    assert engine.frames == before
    engine.save_record()
    assert engine.store.check(True)["valid"]


def test_explicit_motion_gate_stop_and_home(engine):
    record_short(engine, 3)
    with pytest.raises(RuntimeError, match="准备从臂"):
        engine.replay_robot(0, 0.5)
    assert engine.robot.send_count == 0
    engine.set_home()
    with pytest.raises(ValueError):
        engine.prepare(False)
    engine.prepare(True)
    engine.replay_robot(0, 0.5)
    wait_until(lambda: engine.robot.send_count > 3)
    with pytest.raises(RuntimeError, match="正在回放"):
        engine.start_record("must not record replay")
    engine.stop()
    sends = engine.robot.send_count
    time.sleep(0.15)
    assert engine.robot.send_count == sends
    engine.go_home()
    wait_until(lambda: engine.motion is None, timeout=6)
    np.testing.assert_allclose(engine.robot.target, engine.home, atol=1)


def test_stale_arm_feedback_stops_motion(engine, monkeypatch):
    engine.set_home()
    engine.prepare(True)
    engine.go_home()
    wait_until(lambda: engine.robot.send_count > 2)
    original = engine.robot.read
    def stale():
        sample = original()
        sample.feedback_age = 9
        return sample
    monkeypatch.setattr(engine.robot, "read", stale)
    wait_until(lambda: engine.motion is None)
    assert not engine.prepared
    assert "反馈缺失或超时" in engine.error


def test_disconnect_closes_draft_for_later_recovery(engine):
    engine.start_record("interrupted episode")
    wait_until(lambda: engine.frames >= 2)
    engine.disconnect()
    assert engine.phase == "draft"
    assert engine.draft is not None
    engine.save_record()
    assert engine.store.check(True)["valid"]
