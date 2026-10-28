"""Streaming LeRobot v2.1 writer with embedded RGB images and recoverable commits.

Training schema deliberately matches LeoApollo/piper_single/piper_dataset.
Capture diagnostics and drafts live beside the dataset, never in its schema.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock
from PIL import Image

CAMERAS = ("image", "wrist_image")
SCALARS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
FEATURES = {
    **{k: {"dtype": "image", "shape": [480, 640, 3],
            "names": ["height", "width", "channel"]} for k in CAMERAS},
    "state": {"dtype": "float32", "shape": [7], "names": ["state"]},
    "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
    **{k: {"dtype": "float32" if k == "timestamp" else "int64",
            "shape": [1], "names": None} for k in SCALARS},
}


def schema() -> pa.Schema:
    fields = [pa.field(k, pa.struct([("bytes", pa.binary()), ("path", pa.string())])) for k in CAMERAS]
    fields += [pa.field(k, pa.list_(pa.float32(), 7)) for k in ("state", "actions")]
    fields += [pa.field(k, pa.float32() if k == "timestamp" else pa.int64()) for k in SCALARS]
    hf = {k: {"_type": "Image"} for k in CAMERAS}
    hf.update({k: {"_type": "Sequence", "length": 7,
                    "feature": {"_type": "Value", "dtype": "float32"}} for k in ("state", "actions")})
    hf.update({k: {"_type": "Value", "dtype": "float32" if k == "timestamp" else "int64"} for k in SCALARS})
    return pa.schema(fields, metadata={b"huggingface": json.dumps({"info": {"features": hf}}).encode()})


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []


class Draft:
    """Bounded producer queue; RGB frames are spooled to disk while recording."""
    def __init__(self, support: Path, task: str, context: dict):
        self.path = support / "drafts" / uuid.uuid4().hex
        self.path.mkdir(parents=True)
        self.task = task
        self.count = 0
        self.error: str | None = None
        self.closed = False
        self._queue = queue.Queue(maxsize=32)
        write_json(self.path / "draft.json", {"draft_id": self.path.name, "task": task, "context": context})
        self._worker = threading.Thread(target=self._run, daemon=True, name="episode-disk-writer")
        self._worker.start()

    def put(self, frame: dict) -> None:
        if self.error:
            raise RuntimeError(self.error)
        if self.closed:
            raise RuntimeError("当前草稿已关闭")
        try:
            self._queue.put_nowait(frame)
        except queue.Full as exc:
            raise RuntimeError("磁盘写入队列已满，录制已暂停；请保存当前草稿") from exc

    def _run(self):
        try:
            with (self.path / "frames.jsonl").open("a", encoding="utf-8") as f:
                while True:
                    item = self._queue.get()
                    try:
                        if item is None:
                            break
                        i = self.count
                        for cam in CAMERAS:
                            rgb = item[cam]
                            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                                raise ValueError(f"{cam} 必须是 uint8 RGB 480×640×3")
                            Image.fromarray(rgb).save(self.path / f"{i:06d}_{cam}.png", compress_level=1)
                        row = {k: v for k, v in item.items() if k not in CAMERAS}
                        row["draft_index"] = i
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        f.flush()
                        self.count += 1
                    finally:
                        self._queue.task_done()
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            self.error = f"草稿写入失败：{exc}"

    def close(self):
        if not self.closed:
            self.closed = True
            # Never block forever if the disk worker has already failed.
            while self._worker.is_alive():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._worker.join(timeout=30)
            if self._worker.is_alive():
                raise RuntimeError("磁盘写入仍未完成，草稿保留在磁盘，请稍后重试")
        if self.error:
            raise RuntimeError(self.error)


class DatasetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.support = self.root.parent / f".{self.root.name}.studio"
        self.support.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.support / "dataset.lock"), timeout=60)
        with self.lock:
            self._recover_commit()
            self.info = self._read_info()
        self._cached_file = None
        self._cached_group = None
        self._cached_table = None

    def _read_info(self):
        p = self.root / "meta/info.json"
        if not p.exists():
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("目标目录非空且不是支持的数据集；请换一个新目录")
            return None
        info = json.loads(p.read_text())
        if info.get("codebase_version") != "v2.1" or info.get("features") != FEATURES or info.get("fps") != 10:
            raise ValueError("仅支持与样例一致的 LeRobot v2.1 / 10 FPS / image、wrist_image、state、actions 数据集")
        if info.get("robot_type") != "piper":
            raise ValueError("robot_type 必须为 piper")
        return info

    def episodes(self) -> list[dict]:
        with self.lock:
            return read_jsonl(self.root / "meta/episodes.jsonl")

    def ensure_capture_source(self, mode: str):
        """Prevent a simulator run from silently contaminating training data."""
        kind = "sim" if mode == "sim" else "hardware"
        path = self.support / "capture_source.json"
        with self.lock:
            if path.exists():
                if json.loads(path.read_text())["kind"] != kind:
                    raise ValueError("模拟数据与真机数据不能写入同一目录，请修改 dataset_root")
            elif self.info and kind == "sim":
                raise ValueError("不能向来源未知的已有数据集追加模拟数据，请新建模拟数据目录")
            else:
                write_json(path, {"kind": kind})

    def drafts(self) -> list[dict]:
        result = []
        for p in sorted((self.support / "drafts").glob("*/draft.json")):
            d = json.loads(p.read_text())
            rows = self._draft_rows(p.parent)
            result.append({"id": p.parent.name, "task": d["task"], "frames": len(rows)})
        return result

    def _draft_path(self, draft_id: str) -> Path:
        if not draft_id or any(c not in "0123456789abcdef" for c in draft_id):
            raise ValueError("无效草稿 ID")
        p = self.support / "drafts" / draft_id
        if not (p / "draft.json").exists():
            raise ValueError("草稿不存在")
        return p

    @staticmethod
    def _draft_rows(path: Path) -> list[dict]:
        rows = []
        p = path / "frames.jsonl"
        if not p.exists():
            return rows
        lines = p.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    break  # An interrupted final write is not a complete frame.
                raise
            if not all((path / f"{row['draft_index']:06d}_{cam}.png").is_file() for cam in CAMERAS):
                raise ValueError("草稿相机图像缺失，不能静默填充")
            rows.append(row)
        return rows

    def discard(self, draft_id: str):
        # Recoverable discard; never touches saved episodes or original datasets.
        p = self._draft_path(draft_id)
        trash = self.support / "discarded"
        trash.mkdir(exist_ok=True)
        p.rename(trash / p.name)

    def _recover_commit(self):
        marker = self.support / "pending_commit.json"
        if not marker.exists():
            return
        transaction = json.loads(marker.read_text())
        stage = self.support / transaction["stage"]
        for name in transaction["files"]:
            src = stage / name
            dst = self.root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                os.replace(src, dst)
            elif not dst.exists():
                raise RuntimeError(f"提交恢复失败，缺少 {name}；原始草稿仍被保留")
        draft = self.support / "drafts" / transaction["draft_id"]
        episode = read_jsonl(self.root / "meta/episodes.jsonl")[transaction["episode_index"]]
        write_json(self.support / "receipts" / f"{transaction['draft_id']}.json", episode)
        if draft.exists():
            archive = self.support / "committed"
            archive.mkdir(exist_ok=True)
            # Keep scalar acquisition diagnostics; remove only redundant staged images.
            for name in ("frames.jsonl", "draft.json"):
                if (draft / name).exists():
                    dst = archive / f"episode_{transaction['episode_index']:06d}_{name}"
                    shutil.copy2(draft / name, dst)
            shutil.rmtree(draft)
        marker.unlink()
        if stage.exists():
            shutil.rmtree(stage)

    def commit(self, draft_id: str) -> dict:
        if not draft_id or any(c not in "0123456789abcdef" for c in draft_id):
            raise ValueError("无效草稿 ID")
        with self.lock:
            self._recover_commit()
            info = self._read_info()
            self.info = info
            receipt = self.support / "receipts" / f"{draft_id}.json"
            if receipt.exists():
                return json.loads(receipt.read_text())
            path = self._draft_path(draft_id)
            rows = self._draft_rows(path)
            if not rows:
                raise ValueError("草稿为空，无法保存")
            task = json.loads((path / "draft.json").read_text())["task"]
            episodes = read_jsonl(self.root / "meta/episodes.jsonl")
            tasks = read_jsonl(self.root / "meta/tasks.jsonl")
            stats = read_jsonl(self.root / "meta/episodes_stats.jsonl")
            episode_index = len(episodes)
            if info and (info["total_episodes"] != episode_index or len(stats) != episode_index):
                raise ValueError("已有数据集的元数据不一致，停止追加")
            base_index = sum(e["length"] for e in episodes)
            task_index = next((t["task_index"] for t in tasks if t["task"] == task), len(tasks))
            if task_index == len(tasks):
                tasks.append({"task_index": task_index, "task": task})
            stage = self.support / ("txn-" + uuid.uuid4().hex)
            (stage / "meta").mkdir(parents=True)
            rel = f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
            data_path = stage / rel
            data_path.parent.mkdir(parents=True)
            numeric = {k: [] for k in ("state", "actions", *SCALARS)}
            img_stats = {k: {"sum": np.zeros(3), "sumsq": np.zeros(3), "min": np.ones(3),
                             "max": np.zeros(3), "pixels": 0} for k in CAMERAS}
            batch = []
            try:
                with pq.ParquetWriter(data_path, schema(), compression="snappy") as writer:
                    for i, r in enumerate(rows):
                        out = {"state": r["state"], "actions": r["actions"], "timestamp": i / 10,
                               "frame_index": i, "episode_index": episode_index,
                               "index": base_index + i, "task_index": task_index}
                        for k in ("state", "actions"):
                            a = np.asarray(out[k], dtype=np.float32)
                            if a.shape != (7,) or not np.isfinite(a).all():
                                raise ValueError(f"帧 {i} 的 {k} 不是有限的 7 维向量")
                            out[k] = a.tolist()
                        for k in numeric:
                            numeric[k].append(out[k])
                        for cam in CAMERAS:
                            p = path / f"{r['draft_index']:06d}_{cam}.png"
                            out[cam] = {"bytes": p.read_bytes(), "path": None}
                            with Image.open(p) as image:
                                if image.size != (640, 480) or image.mode != "RGB":
                                    raise ValueError(f"帧 {i} 的 {cam} 不是 640×480 RGB")
                                sample = np.asarray(image)[::8, ::8].reshape(-1, 3).astype(np.float64) / 255
                            s = img_stats[cam]
                            s["sum"] += sample.sum(0)
                            s["sumsq"] += (sample * sample).sum(0)
                            s["min"] = np.minimum(s["min"], sample.min(0))
                            s["max"] = np.maximum(s["max"], sample.max(0))
                            s["pixels"] += len(sample)
                        batch.append(out)
                        if len(batch) == 8:
                            writer.write_table(pa.Table.from_pylist(batch, schema=schema()))
                            batch.clear()
                    if batch:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema()))
                episode_stats = {}
                for k, values in numeric.items():
                    a = np.asarray(values, dtype=np.float64)
                    if a.ndim == 1:
                        a = a[:, None]
                    episode_stats[k] = {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
                                        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [len(rows)]}
                for k, s in img_stats.items():
                    mean = s["sum"] / s["pixels"]
                    std = np.sqrt(np.maximum(s["sumsq"] / s["pixels"] - mean * mean, 0))
                    episode_stats[k] = {n: v.reshape(3, 1, 1).tolist() for n, v in
                                         (("min", s["min"]), ("max", s["max"]), ("mean", mean), ("std", std))}
                    episode_stats[k]["count"] = [len(rows)]
                episode = {"episode_index": episode_index, "tasks": [task], "length": len(rows)}
                episodes.append(episode)
                stats.append({"episode_index": episode_index, "stats": episode_stats})
                new_info = {"codebase_version": "v2.1", "robot_type": "piper",
                            "total_episodes": len(episodes), "total_frames": base_index + len(rows),
                            "total_tasks": len(tasks), "total_videos": 0,
                            "total_chunks": episode_index // 1000 + 1, "chunks_size": 1000,
                            "fps": 10, "splits": {"train": f"0:{len(episodes)}"},
                            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                            "features": FEATURES}
                write_jsonl(stage / "meta/tasks.jsonl", tasks)
                write_jsonl(stage / "meta/episodes.jsonl", episodes)
                write_jsonl(stage / "meta/episodes_stats.jsonl", stats)
                write_json(stage / "meta/info.json", new_info)
                write_json(self.support / "pending_commit.json", {
                    "stage": stage.name, "draft_id": draft_id, "episode_index": episode_index,
                    "files": [rel, "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/episodes_stats.jsonl", "meta/info.json"]})
                self._recover_commit()
                self.info = new_info
                return episode
            except Exception:
                # Keep stage + original draft. A prepared manifest can be recovered on next open.
                raise

    def episode_path(self, index: int) -> Path:
        episodes = self.episodes()
        if index < 0 or index >= len(episodes):
            raise ValueError("episode 不存在")
        return self.root / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"

    def signals(self, index: int) -> dict:
        return pq.read_table(self.episode_path(index), columns=["state", "actions", *SCALARS]).to_pydict()

    def frame(self, index: int, frame_index: int) -> dict:
        p = self.episode_path(index)
        pf = pq.ParquetFile(p)
        if not 0 <= frame_index < pf.metadata.num_rows:
            raise ValueError("帧索引越界")
        offset = 0
        for g in range(pf.num_row_groups):
            count = pf.metadata.row_group(g).num_rows
            if frame_index < offset + count:
                if self._cached_file != p or self._cached_group != g:
                    self._cached_table = pf.read_row_group(g)
                    self._cached_file, self._cached_group = p, g
                row = self._cached_table.slice(frame_index - offset, 1).to_pylist()[0]
                for cam in CAMERAS:
                    image = row[cam]
                    if image.get("bytes") is not None:
                        row[cam] = np.asarray(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
                    else:
                        # Existing datasets may store relative paths; constrain them to the dataset.
                        target = (self.root / image["path"]).resolve()
                        if not target.is_relative_to(self.root):
                            raise ValueError("图像路径超出数据集目录")
                        row[cam] = np.asarray(Image.open(target).convert("RGB"))
                return row
            offset += count
        raise ValueError("未找到帧")

    def check(self, images: bool = False) -> dict:
        episodes = self.episodes()
        total, pairs, matches = 0, 0, 0
        problems = []
        tasks = read_jsonl(self.root / "meta/tasks.jsonl")
        for e in episodes:
            data = self.signals(e["episode_index"])
            n = len(data["state"])
            s, a = np.asarray(data["state"]), np.asarray(data["actions"])
            if s.shape != (n, 7) or a.shape != (n, 7) or not np.isfinite(s).all() or not np.isfinite(a).all():
                problems.append(f"episode {e['episode_index']}：非有限值或维度错误")
            if data["frame_index"] != list(range(n)) or data["index"] != list(range(total, total + n)):
                problems.append(f"episode {e['episode_index']}：帧索引错误")
            if n != e["length"] or not np.allclose(data["timestamp"], np.arange(n) / 10, atol=1e-5):
                problems.append(f"episode {e['episode_index']}：长度或时间戳错误")
            if any(x >= len(tasks) or x < 0 for x in data["task_index"]):
                problems.append(f"episode {e['episode_index']}：task_index 错误")
            if any(x != e["episode_index"] for x in data["episode_index"]):
                problems.append(f"episode {e['episode_index']}：episode_index 错误")
            pairs += max(n - 1, 0)
            matched = int(np.all(s[1:] == a[:-1], axis=1).sum()) if n > 1 else 0
            matches += matched
            if matched != max(n - 1, 0):
                problems.append(f"episode {e['episode_index']}：state 与上一帧 actions 不一致")
            total += n
            if images:
                for j in range(n):
                    f = self.frame(e["episode_index"], j)
                    for cam in CAMERAS:
                        if f[cam].shape != (480, 640, 3) or f[cam].dtype != np.uint8:
                            problems.append(f"episode {e['episode_index']} 帧 {j}：图像格式错误")
        info = self._read_info()
        if info and (total != info["total_frames"] or len(episodes) != info["total_episodes"]):
            problems.append("info.json 总数与数据不符")
        if len(read_jsonl(self.root / "meta/episodes_stats.jsonl")) != len(episodes):
            problems.append("episodes_stats.jsonl 数量不符")
        return {"valid": not problems, "episodes": len(episodes), "frames": total,
                "adjacent_pairs": pairs, "previous_action_matches": matches, "problems": problems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--images", action="store_true", help="同时逐帧解码两路图像")
    args = parser.parse_args()
    result = DatasetStore(args.root).check(args.images)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
