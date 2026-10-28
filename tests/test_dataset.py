import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from piper_studio.dataset import DatasetStore, FEATURES, read_jsonl


def test_format_matches_reference_and_hf_decodes_rgb(store, make_draft):
    from datasets import Dataset
    draft = make_draft()
    store.commit(draft.path.name)
    reference = json.loads((Path(__file__).parent / "fixtures/reference_info.json").read_text())
    for key in ("codebase_version", "robot_type", "chunks_size", "fps", "features", "data_path", "video_path"):
        assert store.info[key] == reference[key]
    assert store.info["total_frames"] == 12
    assert store.info["total_videos"] == 0
    data = Dataset.from_parquet(str(store.episode_path(0)))
    row = data[0]
    assert list(row) == list(FEATURES)
    assert data.features["state"].feature.dtype == "float32"
    assert data.features["state"].length == 7
    assert data.features["timestamp"].dtype == "float32"
    assert data.features["index"].dtype == "int64"
    assert np.array(row["image"]).shape == (480, 640, 3)
    np.testing.assert_array_equal(np.array(row["image"])[0, 0], [30, 100, 210])
    assert row["actions"][6] == -70  # Preserve raw encoder noise; don't normalize saved values.
    check = store.check(images=True)
    assert check["valid"], check
    assert check["previous_action_matches"] == check["adjacent_pairs"] == 11
    stats = read_jsonl(store.root / "meta/episodes_stats.jsonl")[0]["stats"]
    assert np.asarray(stats["image"]["mean"]).shape == (3, 1, 1)
    assert stats["image"]["count"] == [12]
    np.testing.assert_allclose(np.ravel(stats["image"]["mean"]), [35.5 / 255, 100 / 255, 204.5 / 255])


def test_append_restart_tasks_and_row_group_seek(store, make_draft):
    store.commit(make_draft(12, "Task A").path.name)
    store.commit(make_draft(3, "Task B").path.name)
    resumed = DatasetStore(store.root)
    resumed.commit(make_draft(2, "Task A").path.name)
    assert [e["length"] for e in resumed.episodes()] == [12, 3, 2]
    assert resumed.signals(1)["index"] == [12, 13, 14]
    assert resumed.signals(2)["task_index"] == [0, 0]
    assert resumed.info["total_tasks"] == 2
    assert resumed.frame(0, 10)["image"][0, 0, 0] == 40
    assert resumed.frame(0, 1)["image"][0, 0, 0] == 31
    assert resumed.frame(1, 0)["episode_index"] == 1
    assert resumed.check(True)["valid"]


def test_commit_recovers_interrupted_metadata_replace(store, make_draft, monkeypatch):
    import piper_studio.dataset as module
    draft = make_draft(3)
    original = module.os.replace
    def interrupted(src, dst):
        if Path(dst) == store.root / "meta/info.json":
            raise OSError("simulated power loss before final metadata publish")
        return original(src, dst)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", interrupted)
        with pytest.raises(OSError):
            store.commit(draft.path.name)
    assert draft.path.exists()
    recovered = DatasetStore(store.root)
    assert recovered.info["total_episodes"] == 1
    assert recovered.check(True)["valid"]
    assert not draft.path.exists()
    assert not (store.support / "pending_commit.json").exists()
    assert (store.support / "committed/episode_000000_frames.jsonl").exists()
    assert recovered.commit(draft.path.name)["episode_index"] == 0
    assert recovered.info["total_episodes"] == 1  # Retrying a committed save never duplicates it.


def test_retry_failed_save_in_same_process_recovers_exactly_once(store, make_draft, monkeypatch):
    import piper_studio.dataset as module
    draft = make_draft(3)
    original = module.os.replace
    def interrupted(src, dst):
        if Path(dst) == store.root / "meta/episodes.jsonl":
            raise OSError("simulated interruption")
        return original(src, dst)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", interrupted)
        with pytest.raises(OSError):
            store.commit(draft.path.name)
    assert store.commit(draft.path.name)["length"] == 3
    assert store.info["total_episodes"] == 1
    assert store.check(True)["valid"]


def test_incomplete_last_draft_line_is_recoverable(store, make_draft):
    draft = make_draft(2)
    with (draft.path / "frames.jsonl").open("a") as f:
        f.write('{"draft_index":2,"state":')
    assert store.commit(draft.path.name)["length"] == 2


def test_missing_image_is_rejected_not_silently_filled(store, make_draft):
    draft = make_draft(2)
    (draft.path / "000001_wrist_image.png").unlink()
    with pytest.raises(ValueError, match="图像缺失"):
        store.commit(draft.path.name)
    assert draft.path.exists()


def test_sim_hardware_separation_and_recoverable_discard(store, make_draft):
    store.ensure_capture_source("sim")
    with pytest.raises(ValueError, match="不能写入同一目录"):
        store.ensure_capture_source("hardware_bus")
    draft = make_draft(2)
    store.discard(draft.path.name)
    assert (store.support / "discarded" / draft.path.name / "frames.jsonl").exists()
    assert store.episodes() == []


def test_refuse_non_dataset_directory(tmp_path):
    (tmp_path / "important.txt").write_text("original")
    with pytest.raises(ValueError, match="目标目录非空"):
        DatasetStore(tmp_path)
    assert (tmp_path / "important.txt").read_text() == "original"
