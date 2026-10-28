"""Optional independent compatibility check, using the official LeRobot 0.3.3 reader.

Run in an existing LeRobot v2.1 environment; this is not a runtime dependency.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
dataset = LeRobotDataset(repo_id="local/piper_dataset", root=args.root.resolve(), download_videos=False,
                         delta_timestamps={"actions": [0.0, 0.1, 0.2, 0.3]})
for index in {0, min(10, len(dataset) - 1), len(dataset) - 1}:
    frame = dataset[index]
    assert tuple(frame["image"].shape) == (3, 480, 640)
    assert tuple(frame["wrist_image"].shape) == (3, 480, 640)
    assert tuple(frame["state"].shape) == (7,)
    assert tuple(frame["actions"].shape) == (4, 7)
    assert np.isfinite(frame["actions"].numpy()).all()
print(json.dumps({"reader": "official LeRobotDataset", "episodes": dataset.num_episodes,
                  "frames": len(dataset), "fps": dataset.fps,
                  "image_tensor": [3, 480, 640], "state_tensor": [7], "action_chunk_tensor": [4, 7],
                  "valid": True}, indent=2))
