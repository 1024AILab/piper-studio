from pathlib import Path
import time

import numpy as np
import pytest

from piper_studio.config import Config
from piper_studio.dataset import DatasetStore, Draft
from piper_studio.engine import Engine


def wait_until(predicate, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for engine condition")


@pytest.fixture
def store(tmp_path):
    return DatasetStore(tmp_path / "piper_dataset")


@pytest.fixture
def make_draft(store):
    created = []
    def make(n=12, task="Move the cube"):
        draft = Draft(store.support, task, {"mode": "test"})
        created.append(draft)
        previous = None
        for i in range(n):
            a = [i * 100, 30000 + i * 60, -50000 - i * 40, 0, 20000, 0, -70 + i * 4000]
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[:, :, 0], image[:, :, 1], image[:, :, 2] = 30 + i, 100, 210 - i
            draft.put({"image": image, "wrist_image": image[:, ::-1].copy(),
                       "state": previous or a, "actions": a, "acquired_monotonic": 10 + i / 10})
            previous = a
        draft.close()
        return draft
    yield make
    for d in created:
        if not d.closed:
            d.close()


@pytest.fixture
def engine(tmp_path):
    e = Engine(Config(dataset_root=str(tmp_path / "sim_dataset")))
    e.connect()
    wait_until(lambda: e.sample is not None and all(c.read()[0] is not None for c in e.cameras.values()))
    yield e
    e.close()
