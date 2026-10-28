from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

import numpy as np

from .config import Config
from .dataset import CAMERAS, DatasetStore, Draft, write_json
from .devices import Camera, PiperHardware, SimRobot
from .motion import Trajectory, limited_step, validate_targets


class Engine:
    """One local operator, one capture worker. UI polling never drives the robot."""
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.store = DatasetStore(config.dataset_root)
        self.lock = threading.RLock()
        self.operation_lock = threading.Lock()
        self.preview_lock = threading.Lock()
        self.robot = None
        self.cameras = {}
        self.connected = False
        self.stop_event = threading.Event()
        self.worker = None
        self.sample = None
        self.phase = "idle"
        self.draft = None
        self.frames = 0
        self.record_start = 0.0
        self.next_record = 0.0
        self.previous_action = None
        self.last_camera_stamps = {}
        self.last_record_at = None
        self.observed_fps = 0.0
        self.prepared = False
        self.teleop = False
        self.last_sent = None
        self.motion = None
        self.motion_started = 0.0
        self.motion_name = ""
        self.motion_frame = 0
        self.home = None
        self.error = ""
        self.logs = collections.deque(maxlen=80)
        self.trace = collections.deque(maxlen=150)
        self.preview_index = None
        self.preview_frame = 0
        self.preview_playing = False
        self.preview_started = 0.0
        self.preview_from = 0
        self.preview_speed = 1.0
        self.preview_signals = None
        self.log("平台已启动。模拟数据和真机数据请使用不同目录。")

    def log(self, message):
        self.logs.append(f"{time.strftime('%H:%M:%S')}  {message}")

    def configure(self, config: Config):
        config.validate()
        with self.operation_lock:
            with self.lock:
                if self.draft or self.phase == "saving" or self.motion:
                    raise RuntimeError("请先保存或丢弃当前草稿，并结束机械臂回放")
            new_store = DatasetStore(config.dataset_root)
            self.disconnect()
            with self.lock:
                self.config, self.store = config, new_store
                self.preview_index = self.preview_signals = None
                self.preview_playing = False
                self.home = None
                self.error = ""
                self.log(f"配置已应用：{config.mode} · {self.store.root}")

    def connect(self):
        with self.operation_lock:
            if self.connected:
                return "设备已连接"
            robot = SimRobot(self.config) if self.config.mode == "sim" else PiperHardware(self.config)
            cameras = {"image": Camera(self.config.image), "wrist_image": Camera(self.config.wrist_image, wrist=True)}
            try:
                robot.connect()
                for camera in cameras.values():
                    camera.connect()
                with self.lock:
                    self.robot, self.cameras = robot, cameras
                    self.connected = True
                    self.prepared = self.teleop = False
                    self.error = ""
                    self.stop_event.clear()
                    self.worker = threading.Thread(target=self._loop, daemon=True, name="piper-acquisition")
                    self.worker.start()
                    self.log("设备已连接。真机连接阶段没有发送使能、失能、回零或运动指令。")
                return "已连接；等待新鲜的关节和相机数据"
            except Exception:
                for camera in cameras.values():
                    camera.close()
                robot.close()
                raise

    def disconnect(self):
        with self.lock:
            if self.phase in {"armed", "recording"}:
                self._pause("连接断开，当前录制已暂停")
            self._stop_motion(hold=True)
            self.stop_event.set()
        if self.worker and self.worker is not threading.current_thread():
            self.worker.join(timeout=3)
            if self.worker.is_alive():
                raise RuntimeError("采集线程未退出；不能关闭仍在使用的设备")
        if self.robot:
            self.robot.close()
        for camera in self.cameras.values():
            camera.close()
        with self.lock:
            self.connected = False
            self.prepared = False
            self.robot = None
            self.cameras = {}
            self.log("已断开设备通信；没有失能机械臂。")

    def _pause(self, reason):
        self.phase = "draft" if self.draft else "idle"
        self.error = reason
        self.log(reason)

    def _stop_motion(self, hold=True):
        active = self.motion is not None or self.teleop
        self.motion = None
        self.teleop = False
        if active and hold and self.robot:
            try:
                self.robot.hold()
            except Exception as exc:
                self.log(f"停止发送；保持位置命令未完成：{exc}")
        if active:
            self.log("平台运动已停止；硬件主从直连不由此按钮中断。")

    def stop(self):
        with self.lock:
            self._stop_motion()
            if self.phase in {"armed", "recording"}:
                self._pause("录制已停止，草稿待保存")
            self.preview_playing = False
        return "已停止平台回放/遥操；当前草稿保留。硬件主从仍需使用设备自身的停止方式。"

    def _target_status(self, allow_hold=False):
        s = self.sample
        labels = ("J1 / J2", "J3 / J4", "J5 / J6", "夹爪")
        groups = ((0, 1), (2, 3), (4, 5), (6,))
        parts = []
        if s and s.action_frame_ages:
            for index, (identifier, age) in enumerate(s.action_frame_ages.items()):
                parts.append({"label": labels[index], "id": identifier, "age": age,
                              "channels": groups[index]})
        else:
            parts = [{"label": "关节 / 夹爪", "id": "", "channels": tuple(range(7)),
                      "age": s.action_age if s and s.action is not None else None}]
        unavailable = [p for p in parts if p["age"] is None or p["age"] > self.config.stale_after_s]
        complete = s is not None and s.action is not None and np.isfinite(s.action).all()
        held = []
        if complete and not unavailable:
            return {"ready": True, "held_channels": [], "parts": parts, "message": "主臂目标实时更新"}
        # A position command remains the last received setpoint while the pendant
        # is quiet. Accept it only during an already-started hardware recording,
        # with fresh, enabled feedback at that setpoint AND a stationary history.
        # Never initialize targets from feedback or relax software motion health.
        if (allow_hold and complete and self.phase == "recording" and self.previous_action is not None
                and self.config.mode == "hardware_bus" and self.config.leader_source == "control"
                and self.config.hold_stationary_targets and s.enabled and not s.error
                and s.follower is not None and s.feedback_age <= self.config.stale_after_s
                and all(p["age"] is not None for p in unavailable)):
            channels = sorted({i for p in unavailable for i in p["channels"]})
            now = time.monotonic()
            history = [(at, q) for at, q, _ in self.trace if now - at <= 0.25]
            if history and history[-1][0] - history[0][0] >= 0.15 and now - history[-1][0] <= 0.1:
                positions = np.vstack([q for _, q in history] + [s.follower])
                stationary_limit = np.asarray([100] * 6 + [300])  # 0.1° / 0.3 mm over >=150 ms.
                if (np.array_equal(s.action[channels], self.previous_action[channels])
                        and np.all(np.abs(s.follower[channels] - s.action[channels]) <= 1000)
                        and np.all(np.ptp(positions[:, channels], axis=0) <= stationary_limit[channels])):
                    held = channels
        if held:
            message = "静止等待：保留已收到且从臂已到达的目标；新目标到达后继续更新"
        else:
            names = "、".join(p["label"] for p in unavailable) or "关节 / 夹爪"
            message = f"等待主臂的 {names} 目标；请通过示教器开始操作"
        return {"ready": bool(held), "held_channels": held, "parts": parts, "message": message}

    def _health(self, need_action=False, need_cameras=False, allow_held_target=False):
        target = None
        if not self.connected or not self.sample:
            raise RuntimeError("请先连接设备")
        s = self.sample
        if s.error:
            raise RuntimeError(s.error)
        if s.follower is None or s.feedback_age > self.config.stale_after_s:
            raise RuntimeError("从臂反馈缺失或超时")
        if need_action:
            target = self._target_status(allow_hold=allow_held_target)
            if not target["ready"]:
                raise RuntimeError("录制所需的主臂目标未就绪：" + target["message"]
                                   + "。从臂反馈正常也不能代替缺失的目标；已有草稿保留")
        if need_cameras:
            for name, camera in self.cameras.items():
                image, age, _ = camera.read()
                if image is None or age > self.config.camera_stale_after_s or camera.error:
                    raise RuntimeError(f"{name} 相机未就绪：{camera.error or '图像超时'}")
        return target

    def _loop(self):
        deadline = time.monotonic()
        last_tick = deadline
        while not self.stop_event.is_set():
            now = time.monotonic()
            dt = min(max(now - last_tick, 0.001), 0.05)
            last_tick = now
            try:
                sample = self.robot.read()
                with self.lock:
                    self.sample = sample
                    if self.motion or self.teleop:
                        self._health()
                        if not sample.enabled:
                            raise RuntimeError("从臂未使能，停止平台控制")
                        if self.config.mode != "sim" and self.robot.external_control_recent():
                            raise RuntimeError("检测到从臂总线上的外部控制帧，停止平台发送")
                        if self.last_sent is not None and np.max(np.abs(sample.follower[:6] - self.last_sent[:6])) > self.config.max_follow_error_deg * 1000:
                            raise RuntimeError("从臂跟随误差超过阈值，停止平台控制")
                        if self.motion:
                            target, index, done = self.motion.at(now - self.motion_started)
                            # If the loop stalls, catch up gradually rather than jumping.
                            target = limited_step(self.last_sent, target, dt, self.config)
                            sent = self.robot.send(target)
                            self.last_sent, self.motion_frame = sent, index
                            if done and np.max(np.abs(target - self.motion.actions[-1])) < 2:
                                self.motion = None
                                self.log("机械臂轨迹已发送完成，保持末帧目标。")
                        elif self.teleop:
                            self._health(need_action=True, need_cameras=True)
                            target = limited_step(self.last_sent, sample.action, dt, self.config)
                            self.last_sent = self.robot.send(target)
                    if self.phase == "armed":
                        self._health(need_cameras=True)
                        if self._target_status()["ready"]:
                            self._begin_recording(now)
                    if self.phase == "recording" and now >= self.next_record:
                        self._capture(now)
                    if sample.follower is not None:
                        self.trace.append((now, sample.follower.copy(), sample.action.copy() if sample.action is not None else None))
            except Exception as exc:
                with self.lock:
                    if self.phase in {"armed", "recording"}:
                        self._pause(str(exc))
                    self._stop_motion(hold=True)
                    if self.error != str(exc):
                        self.error = str(exc)
                        self.log(str(exc))
                    self.prepared = False
            deadline += 1 / self.config.control_hz
            if deadline < time.monotonic():
                deadline = time.monotonic()
            self.stop_event.wait(max(0, deadline - time.monotonic()))

    def _capture(self, now):
        target_status = self._health(need_action=True, need_cameras=True, allow_held_target=True)
        if self.config.mode == "software" and not self.teleop:
            raise RuntimeError("软件遥操尚未启动")
        if self.last_record_at is not None and now - self.last_record_at > 1.75 / self.config.fps:
            raise RuntimeError("采集循环延迟超过 1.75 帧；录制已暂停，避免伪造固定帧率时间轴")
        images = {}
        stamps = {}
        for name, cam in self.cameras.items():
            rgb, _, stamp = cam.read()
            if stamp <= self.last_camera_stamps.get(name, -1):
                raise RuntimeError(f"{name} 未产生新图像；录制暂停")
            images[name], stamps[name] = rgb.copy(), stamp
        action = self.last_sent.copy() if self.config.mode == "software" else self.sample.action.copy()
        state = action.copy() if self.previous_action is None else self.previous_action.copy()
        self.draft.put({**images, "state": state.tolist(), "actions": action.tolist(),
                        "follower_state": self.sample.follower.tolist(), "acquired_monotonic": now,
                        "camera_monotonic": stamps, "feedback_age_s": self.sample.feedback_age,
                        "action_age_s": self.sample.action_age,
                        "action_frame_age_s": self.sample.action_frame_ages,
                        "held_target_channels": target_status["held_channels"]})
        self.previous_action = action
        self.last_camera_stamps = stamps
        self.frames += 1
        if self.last_record_at is not None:
            instantaneous = 1 / (now - self.last_record_at)
            self.observed_fps = instantaneous if self.observed_fps == 0 else 0.8 * self.observed_fps + 0.2 * instantaneous
        self.last_record_at = now
        self.next_record += 1 / self.config.fps
        if self.frames >= self.config.max_episode_s * self.config.fps:
            self._pause("已达到单条最长时长，请保存当前草稿")

    def start_record(self, task):
        task = str(task).strip()
        if not task:
            raise ValueError("请先填写采集任务名称")
        with self.lock:
            if self.draft or self.phase == "saving":
                raise RuntimeError("请先保存或丢弃当前草稿")
            if self.motion:
                raise RuntimeError("机械臂正在回放或复位，不能开始采集")
            self._health(need_action=self.config.mode != "hardware_bus", need_cameras=True)
            if self.config.mode == "software" and not self.teleop:
                raise RuntimeError("请先准备从臂并启动软件遥操")
            self.store.ensure_capture_source(self.config.mode)
            self.draft = Draft(self.store.support, task, {"config": self.config.to_dict(),
                                "state_semantics": "previous_recorded_action; first_frame=current_action",
                                "joint_unit": "0.001_degree", "gripper_unit": "micrometer"})
            self.frames = 0
            self.previous_action = None
            self.last_camera_stamps = {}
            self.last_record_at = None
            self.observed_fps = 0
            self.record_start = self.next_record = 0.0
            self.phase = "armed"
            self.error = ""
            if self._target_status()["ready"]:
                self._begin_recording(time.monotonic())
            else:
                self.log(f"已准备录制，等待示教：{task}")
            waiting = self.phase == "armed"
        if waiting:
            return "已准备录制。现在可以操作示教器；收到完整主臂目标后自动开始计帧，无需再次点击。"
        return "正在录制。完成后点击保存；回程和摆物体请放在录制区间之外。"

    def _begin_recording(self, now):
        self.record_start = self.next_record = now
        self.phase = "recording"
        self.error = ""
        self.log(f"开始录制：{self.draft.task}")

    def save_record(self):
        with self.operation_lock:
            with self.lock:
                if not self.draft:
                    raise RuntimeError("没有可保存的草稿")
                if not self.frames:
                    raise RuntimeError("尚未收到可录制的完整数据；请开始示教，或取消等待")
                self.phase = "saving"
                draft = self.draft
            try:
                draft.close()
                episode = self.store.commit(draft.path.name)
            except Exception as exc:
                with self.lock:
                    self.phase = "draft"
                    self.error = str(exc)
                    self.log(f"保存失败，草稿保留：{exc}")
                raise
            with self.lock:
                self.draft = None
                self.phase = "idle"
                self.error = ""
                self.log(f"已保存 episode {episode['episode_index']:03d}，{episode['length']} 帧")
            return f"已保存 episode {episode['episode_index']:03d}，{episode['length']} 帧。现在可人工复位并录下一条。"

    def discard_record(self):
        with self.operation_lock:
            with self.lock:
                if not self.draft:
                    raise RuntimeError("没有当前草稿")
                self.phase = "draft"
                draft = self.draft
            try:
                draft.close()
            except RuntimeError:
                if draft._worker.is_alive():
                    raise
            self.store.discard(draft.path.name)
            with self.lock:
                self.draft = None
                self.phase = "idle"
                self.frames = 0
                self.error = ""
                self.log("当前草稿已移至 discarded，可从磁盘恢复；已保存的数据未修改。")
            return "当前草稿已丢弃，已保存的 episode 未改变"

    def recover_draft(self, draft_id):
        with self.operation_lock:
            with self.lock:
                if self.draft or self.phase == "saving":
                    raise RuntimeError("请先处理当前草稿")
                self.phase = "saving"
            try:
                e = self.store.commit(draft_id)
                self.log(f"恢复并保存草稿为 episode {e['episode_index']}")
                return f"已恢复 episode {e['episode_index']}，{e['length']} 帧"
            finally:
                with self.lock:
                    self.phase = "idle"

    def prepare(self, acknowledged):
        if not acknowledged:
            raise ValueError("真机执行前，请确认主臂已停止向从臂发指令，并检查运动路径")
        with self.lock:
            if self.phase in {"armed", "recording", "saving"} or self.motion:
                raise RuntimeError("录制或运动期间不能准备执行")
            self._health()
            self.robot.prepare()
            self.sample = self.robot.read()
            self.last_sent = self.sample.follower.copy()
            self.prepared = True
            self.log("从臂执行已准备；没有发送 reset 或零点标定指令。")
        return "已准备从臂执行。首次回放会先缓慢靠近首帧。"

    def start_teleop(self):
        with self.lock:
            if self.config.mode == "hardware_bus":
                raise RuntimeError("硬件直连已由主臂控制，无需软件转发")
            if not self.prepared or self.motion or self.phase in {"armed", "recording", "saving"}:
                raise RuntimeError("请先准备从臂，且确保没有正在录制或回放")
            self._health(need_action=True, need_cameras=True)
            if self.config.mode == "sim":
                self.robot.demo = True
                self.teleop = False
            else:
                self.teleop = True
            self.log("遥操已启动")
        return "遥操已启动，可以开始录制"

    def set_home(self):
        with self.lock:
            self._health()
            if self.motion or self.phase in {"armed", "recording"}:
                raise RuntimeError("请在停止录制与回放后记录起始位")
            self.home = self.sample.follower.copy()
            write_json(self.store.support / "home.json", {"raw": self.home.tolist(), "config": self.config.to_dict()})
            self.log("当前从臂姿态已记为本次会话起始位")
        return "已记住当前从臂姿态；这是任务起始位，不是零点标定"

    def go_home(self):
        if self.home is None:
            raise RuntimeError("本次会话尚未记录起始位")
        return self._begin_motion([self.home], 0.5, "回到起始位")

    def _begin_motion(self, actions, speed, name):
        with self.lock:
            if self.draft or self.phase in {"recording", "saving"}:
                raise RuntimeError("请先保存或丢弃录制草稿")
            if not self.prepared:
                raise RuntimeError("请先准备从臂执行")
            if self.motion or self.teleop:
                raise RuntimeError("请先停止当前运动或软件遥操")
            self._health()
            if self.robot.external_control_recent():
                raise RuntimeError("检测到外部主臂控制，禁止同时回放")
            trajectory = Trajectory(self.sample.follower, actions, self.config, float(speed))
            self.last_sent = self.sample.follower.copy()
            self.motion = trajectory
            self.motion_started = time.monotonic()
            self.motion_name = name
            self.motion_frame = 0
            self.log(f"{name}：首帧靠近 {trajectory.approach_s:.1f}s，有效倍率 {trajectory.speed:.2f}×")
        return f"{name}已启动；首帧靠近 {trajectory.approach_s:.1f}s，总计划时间 {trajectory.duration:.1f}s"

    def replay_robot(self, index, speed):
        if index is None:
            raise ValueError("请选择 episode")
        with self.operation_lock:
            signals = self.store.signals(int(index))
            return self._begin_motion(signals["actions"], speed, f"回放 episode {int(index):03d}")

    def load_preview(self, index):
        if index is None:
            raise ValueError("请选择 episode；没有数据时请先完成一次录制")
        with self.preview_lock:
            signals = self.store.signals(int(index))
            self.preview_index = int(index)
            self.preview_frame = 0
            self.preview_playing = False
            self.preview_signals = signals
        return len(signals["actions"])

    def play_preview(self, speed=1.0):
        with self.preview_lock:
            if self.preview_signals is None:
                raise ValueError("请先载入 episode")
            if self.preview_frame >= len(self.preview_signals["actions"]) - 1:
                self.preview_frame = 0
            self.preview_from = self.preview_frame
            self.preview_started = time.monotonic()
            self.preview_speed = float(speed)
            self.preview_playing = True

    def seek_preview(self, frame):
        with self.preview_lock:
            if self.preview_signals is not None:
                self.preview_frame = min(max(int(frame), 0), len(self.preview_signals["actions"]) - 1)
                self.preview_playing = False

    def preview(self):
        with self.preview_lock:
            if self.preview_signals is None:
                return None
            if self.preview_playing:
                self.preview_frame = min(len(self.preview_signals["actions"]) - 1,
                    self.preview_from + int((time.monotonic() - self.preview_started) * 10 * self.preview_speed))
                if self.preview_frame == len(self.preview_signals["actions"]) - 1:
                    self.preview_playing = False
            row = self.store.frame(self.preview_index, self.preview_frame)
            return row

    def snapshot(self):
        with self.lock:
            s = self.sample
            images = {name: cam.read()[0] for name, cam in self.cameras.items()}
            ages = {name: cam.read()[1] for name, cam in self.cameras.items()}
            issues = []
            if self.connected:
                if not s or s.follower is None or s.feedback_age > self.config.stale_after_s:
                    issues.append("关节反馈缺失或超时")
                if s and s.error:
                    issues.append(s.error)
                for name, cam in self.cameras.items():
                    if images[name] is None or ages[name] > self.config.camera_stale_after_s or cam.error:
                        label = f"{name} · {cam.config.device}"
                        issues.append(f"{label}：{cam.error or '等待新鲜图像'}")
            observer = getattr(self.robot, "follower_observer", None)
            observed_ids = observer.frames.recent_ids() if observer else []
            target = self._target_status(allow_hold=True)
            source_ready = self.config.mode == "hardware_bus" or target["ready"]
            if self.config.mode == "software" and not self.teleop:
                source_ready = False
            return {"phase": self.phase, "connected": self.connected, "mode": self.config.mode,
                    "capture_ready": self.connected and not issues and target["ready"],
                    "can_start_record": self.connected and not issues and source_ready,
                    "target_status": target, "can_status": self.can_diagnostics(),
                    "readiness_issues": issues, "observed_can_ids": observed_ids,
                    "frames": self.frames, "fps": self.observed_fps, "prepared": self.prepared,
                    "teleop": self.teleop, "motion": self.motion_name if self.motion else "",
                    "motion_frame": self.motion_frame,
                    "motion_total": len(self.motion.actions) if self.motion else 0,
                    "motion_elapsed": time.monotonic() - self.motion_started if self.motion else 0,
                    "motion_duration": self.motion.duration if self.motion else 0,
                    "sample": s, "images": images, "camera_ages": ages,
                    "error": self.error or (s.error if s and self.connected else ""), "logs": "\n".join(reversed(self.logs)),
                    "trace": list(self.trace), "draft_id": self.draft.path.name if self.draft else None}

    def can_diagnostics(self):
        with self.lock:
            return self.robot.diagnostics() if self.connected and hasattr(self.robot, "diagnostics") else []

    def close(self):
        self.disconnect()
        if self.draft:
            try:
                self.draft.close()
            except RuntimeError:
                pass  # The UI exposes recoverable drafts on next startup.
