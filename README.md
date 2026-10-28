# Piper Studio

**[English](README.md) | [简体中文](README.zh-CN.md)**

A local platform for AgileX PiPER demonstration collection and replay.
Watch two RGB cameras, inspect leader targets and follower feedback, record
episodes, and review every frame before training.

![Platform UI / 平台 UI](docs/images/platform-ui.jpg)

**Platform UI / 平台 UI.** Screenshot of the running application using its
built-in simulated devices. Hardware mode uses the same interface with live
camera and CAN data.

For asynchronous VLA inference and continuous action execution, see
[Piper Realtime VLA](https://github.com/1024AILab/piper-realtime-vla).


## Real robot setup

![Real robot setup / 真机平台](docs/images/real-robot-setup.png)

User-provided PiPER setup photograph: duck pick-and-place with a pen holder.


## What it does

- Two 640×480 RGB views: scene and wrist, with freshness checks.
- Leader-command / follower-feedback monitoring and per-episode recording.
- LeRobot v2.1-compatible Parquet output, task labels, statistics and draft recovery.
- Frame-by-frame visual replay, saved starting poses and controlled robot replay.
- Linux SocketCAN and macOS `gs_usb`; a separate RealSense RGB bridge for macOS.

## Quick start

Python **3.12** is recommended. The included macOS camera binaries require
Apple Silicon and CPython 3.12. No robot or model weights are needed for the demo.

```bash
git clone https://github.com/1024AILab/piper-studio.git
cd piper-studio
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m piper_studio.app --config configs/sim.json --connect-sim --lang en --open
```

Open **http://127.0.0.1:7860**. Enter a task name, start recording, save the
episode, then open **数据与回放** to inspect it. Synthetic camera images are
explicitly labelled in the UI and stored separately from hardware recordings.
The platform UI supports English (`--lang en`, default) and Chinese (`--lang zh`). Restart with the desired language; task labels and stored data are preserved.

To inspect the bundled synthetic episode without connecting devices:

```bash
python -m piper_studio.app --config configs/demo.json --open
```

## Data contract

The format follows the
[LeoApollo Piper example](https://huggingface.co/datasets/LeoApollo/piper_single/tree/main/piper_dataset).

| Item | Stored value |
|---|---|
| Sampling | 10 FPS |
| `image`, `wrist_image` | 640×480 RGB, lossless PNG bytes inside Parquet |
| `actions` | Six joint targets in 0.001° plus one gripper target in µm |
| `state` | Previous commanded action; the first frame uses the current command |
| Labels | Per-episode task plus `task_index` |

This `state` convention matches the reference dataset; it is not a synonym
for measured joint feedback. Feedback and timing diagnostics are retained in
the adjacent `.studio` support directory. Two cameras sample independently;
this is not hardware-triggered synchronization. PNG images are embedded in
Parquet, so a separate video directory is not expected.

```text
datasets/piper_dataset/
  data/chunk-000/episode_000000.parquet
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
```

## Connect hardware

```text
Leader + teaching pendant ── CAN ── Follower
                               └── USB-CAN ── Computer
Scene camera + wrist camera ───────── USB ──── Computer
```

For hardware master/follower operation, the platform observes the shared
CAN bus. Connecting and recording do not enable motors or command motion.
Replay and returning to a saved pose are separate, explicitly armed actions.

### macOS

```bash
python -m pip install -e '.[mac]'
cp configs/mac_connected.json configs/mac_connected.local.json
```

Replace the two placeholder camera IDs in the local config with your devices'
USB identifiers. Verify scene/wrist assignment. `gs_usb` requires a compatible
candleLight / Geschwister Schneider adapter; Linux `can0` naming alone does
not establish compatibility.

```bash
export PIPER_SCENE_SERIAL=YOUR_NUMERIC_SCENE_USB_ID
export PIPER_WRIST_SERIAL=YOUR_NUMERIC_WRIST_USB_ID
./启动RealSense彩色流.command
# In another terminal with the virtual environment active:
python -m piper_studio.app --config configs/mac_connected.local.json --open
```

The camera helper may ask for administrator USB access; the web app runs as
the ordinary user. See [the bundled SDK notes](vendor/realsense_macos/README.md)
for architecture, source version, licenses and rebuilding.

### Linux

```bash
python -m pip install -e '.[hardware,realsense]'
cp configs/hardware_bus.json configs/hardware_bus.local.json
python -m piper_studio.app --config configs/hardware_bus.local.json --open
```

Set the actual CAN interface and camera backends/IDs in your local config.
The bus must already be configured for **1 Mbps**.

## Replay and operating limits

Save the episode before changing modes. To let the computer control the
follower, stop the leader from sending commands, verify the motion path, and
use the UI's explicit preparation controls. Start with a slow replay rate.
The platform bounds targets, monitors fresh feedback and stops its own command
stream on faults. The software stop does not replace a hardware emergency stop,
and no collision-avoidance planner is included. A saved task pose does not
recalibrate encoder zero points.

Only one capture/inference process may own a USB-CAN adapter at a time.

## Project map

| Path | Purpose |
|---|---|
| `piper_studio/app.py` | Browser UI |
| `piper_studio/engine.py` | Recording and replay orchestration |
| `piper_studio/dataset.py` | Episode writer, metadata and recovery |
| `piper_studio/devices.py` | CAN, cameras and simulated devices |
| `piper_studio/realsense_bridge.py` | macOS RGB helper |
| `configs/` | Portable examples; put machine settings in `*.local.json` |
| `tests/` | Existing data-format, device and control regression tests |

```bash
python -m pip install -e '.[test,mac]'  # use hardware instead of mac on Linux
python -m pytest -q
```

See [validation](docs/VALIDATION.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
The backup excludes recorded datasets, private configuration, logs, credentials,
virtual environments and model weights. The small bundled episode is synthetic.
