from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from .storage import read_json


def video_check(path: Path, frames: int, shape: list[int], fps: int) -> dict:
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - fps) > 0.01:
            raise ValueError(f"{path}: incorrect video fps")
        for frame in container.decode(stream):
            if [frame.height, frame.width, 3] != shape:
                raise ValueError(f"{path}: incorrect video dimensions")
            if frame.time is None or abs(frame.time - count / fps) > 0.5 / fps:
                raise ValueError(f"{path}: video presentation timestamp mismatch at frame {count}")
            count += 1
    if count != frames:
        raise ValueError(f"{path}: {count} decoded frames, expected {frames}")
    return {"frames": count, "fps": fps, "shape": shape}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_pending(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    table = pq.read_table(directory / "frames.parquet").to_pydict()
    contract = manifest["contract"]
    n, fps = manifest["length"], contract["fps"]
    for key in ("observation.state", "action"):
        array = np.asarray(table[key])
        if array.shape != (n, len(contract["names"])) or not np.isfinite(array).all():
            raise ValueError(f"pending {key}: invalid shape or values")
    if not np.allclose(table["timestamp"], np.arange(n) / fps, atol=1e-5, rtol=0):
        raise ValueError("pending timestamps do not match fps")
    if table["frame_index"] != list(range(n)):
        raise ValueError("pending frame indices are not contiguous")
    for key, shape in contract["cameras"].items():
        video_check(directory / f"{key}.mp4", n, shape, fps)
    return {"length": n, "issues": manifest["issues"]}


def validate_dataset(root: Path, *, decode_videos: bool = True) -> dict:
    root = root.expanduser().resolve()
    info = read_json(root / "meta/info.json")
    if info["codebase_version"] != "v2.1":
        raise ValueError("this validator targets LeRobot v2.1")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    tasks = read_jsonl(root / "meta/tasks.jsonl")
    stats = read_jsonl(root / "meta/episodes_stats.jsonl")
    if [r["task_index"] for r in tasks] != list(range(len(tasks))):
        raise ValueError("task indices must be contiguous")
    if len(tasks) != info["total_tasks"] or len(episodes) != info["total_episodes"]:
        raise ValueError("metadata totals differ from tasks/episodes")
    if [r["episode_index"] for r in stats] != list(range(len(episodes))):
        raise ValueError("episode statistics do not cover the dataset")
    camera_features = {k: v for k, v in info["features"].items() if v["dtype"] == "video"}
    offset = 0
    identical = []
    for index, episode in enumerate(episodes):
        if episode["episode_index"] != index or episode["length"] <= 0:
            raise ValueError("episode indices must be contiguous and episodes nonempty")
        variables = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
        path = root / info["data_path"].format(**variables)
        table = pq.read_table(path).to_pydict()
        n = episode["length"]
        for key in ("observation.state", "action"):
            array = np.asarray(table[key])
            if array.shape != (n, *info["features"][key]["shape"]) or not np.isfinite(array).all():
                raise ValueError(f"episode {index}: {key} shape or values are invalid")
        if np.array_equal(table["observation.state"], table["action"]):
            identical.append(index)
        for key, expected in {
            "frame_index": list(range(n)),
            "episode_index": [index] * n,
            "index": list(range(offset, offset + n)),
        }.items():
            if table[key] != expected:
                raise ValueError(f"episode {index}: non-contiguous {key}")
        if not np.allclose(table["timestamp"], np.arange(n) / info["fps"], atol=1e-5, rtol=0):
            raise ValueError(f"episode {index}: timestamps do not match fps")
        if len(table["task_index"]) != n or any(t < 0 or t >= len(tasks) for t in table["task_index"]):
            raise ValueError(f"episode {index}: invalid task references")
        for key, feature in camera_features.items():
            video = root / info["video_path"].format(video_key=key, **variables)
            if decode_videos:
                video_check(video, n, feature["shape"], info["fps"])
            elif not video.is_file():
                raise FileNotFoundError(video)
        for key, values in stats[index]["stats"].items():
            for metric in ("mean", "std", "min", "max", "count"):
                if not np.isfinite(np.asarray(values[metric])).all():
                    raise ValueError(f"episode {index}: nonfinite {key}/{metric} statistics")
        offset += n
    if offset != info["total_frames"] or info["total_videos"] != len(episodes) * len(camera_features):
        raise ValueError("frame/video totals disagree with episodes")
    return {
        "root": str(root),
        "episodes": len(episodes),
        "frames": offset,
        "videos_decoded": decode_videos,
        "same_frame_action_equals_state_episodes": identical,
        "contract": read_json(root / "meta/capture_contract.json")
        if (root / "meta/capture_contract.json").exists()
        else None,
    }
