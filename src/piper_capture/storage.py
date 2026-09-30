from __future__ import annotations

import fcntl
import json
import math
import os
import queue
import shutil
import threading
import uuid
import zipfile
from contextlib import AbstractContextManager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CaptureConfig
from .timeline import CaptureError, Frame

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def numeric_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def aggregate_stats(episodes: list[dict]) -> dict:
    if not episodes:
        return {}
    result = {}
    for key in episodes[0]:
        entries = [e[key] for e in episodes]
        counts = np.asarray([e["count"][0] for e in entries], dtype=np.float64)
        means = np.asarray([e["mean"] for e in entries])
        variances = np.asarray([e["std"] for e in entries]) ** 2
        weights = (counts / counts.sum()).reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (means * weights).sum(0)
        variance = ((variances + (means - mean) ** 2) * weights).sum(0)
        result[key] = {
            "min": np.min([e["min"] for e in entries], axis=0).tolist(),
            "max": np.max([e["max"] for e in entries], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(counts.sum())],
        }
    return result


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "ultrafast", "tune": "zerolatency"}
        self.stream.codec_context.thread_count = 1
        self.fps, self.frames = fps, 0

    def write(self, rgb: np.ndarray):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.frames
        frame.time_base = Fraction(1, self.fps)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class EpisodeWriter:
    """Bounded image queue, streaming H.264, small in-memory vector records."""

    def __init__(self, directory: Path, config: CaptureConfig, task: str, mock: bool):
        self.directory, self.config, self.task, self.mock = directory, config, task, mock
        directory.mkdir(parents=True)
        write_json(
            directory / "capture.json",
            {
                "config": config.as_dict(),
                "mock": mock,
                "task": task,
                "action_sampling": "previous leader command; held while CAN and follower feedback are healthy",
            },
        )
        self.queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.error: Exception | None = None
        self.count = 0
        self._finished = False
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=True)
        self._thread.start()

    def append(self, frame: Frame):
        if self._finished:
            raise CaptureError("episode writer is already closed")
        if self.error:
            raise CaptureError(f"episode writer failed: {self.error}") from self.error
        try:
            self.queue.put_nowait(frame)
        except queue.Full as exc:
            raise CaptureError("video writer queue is full; episode cannot be accepted") from exc
        self.count += 1

    def finish(self, failure: str | None = None) -> Path:
        if self._finished:
            raise CaptureError("episode has already been finalized")
        self._finished = True
        while self._thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self.error:
            write_json(self.directory / "failure.json", {"error": str(self.error)})
            raise CaptureError(f"episode writer failed; files retained at {self.directory}: {self.error}")
        manifest = read_json(self.directory / "manifest.json")
        if failure:
            manifest["issues"].append(failure)
            write_json(self.directory / "manifest.json", manifest)
        return self.directory

    def _run(self):
        videos = {}
        records = []
        moments: dict = {}
        sequences = {c.name: [] for c in self.config.cameras}
        try:
            for c in self.config.cameras:
                videos[c.key] = VideoWriter(
                    self.directory / f"{c.key}.mp4", c.width, c.height, self.config.fps
                )
                moments[c.key] = {
                    "sum": np.zeros(3),
                    "sumsq": np.zeros(3),
                    "pixels": 0,
                    "min": np.full(3, np.inf),
                    "max": np.full(3, -np.inf),
                }
            with (self.directory / "timing.jsonl").open("w", encoding="utf-8") as timing:
                while True:
                    frame = self.queue.get()
                    if frame is None:
                        break
                    records.append((frame.time, frame.state, frame.action))
                    timing.write(
                        json.dumps(
                            {"frame_index": len(records) - 1, "host_time": frame.time, **frame.timing},
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    for key, rgb in frame.images.items():
                        videos[key].write(rgb)
                        # Sample spatially, keeping all time steps. RGB statistics are in [0, 1].
                        pixels = rgb[::8, ::8].reshape(-1, 3).astype(np.float64) / 255.0
                        stats = moments[key]
                        stats["sum"] += pixels.sum(0)
                        stats["sumsq"] += (pixels**2).sum(0)
                        stats["pixels"] += len(pixels)
                        stats["min"] = np.minimum(stats["min"], pixels.min(0))
                        stats["max"] = np.maximum(stats["max"], pixels.max(0))
                    for name, detail in frame.timing["cameras"].items():
                        sequences[name].append(detail["sequence"])
            self._finalize_records(records, moments, sequences)
        except Exception as exc:
            self.error = exc
        finally:
            for writer in videos.values():
                try:
                    writer.close()
                except Exception as exc:
                    self.error = self.error or exc
            if self.error is not None:
                (self.directory / "manifest.json").unlink(missing_ok=True)

    def _finalize_records(self, records, moments, sequences):
        n = len(records)
        issues = []
        if n < self.config.min_episode_frames:
            issues.append(f"episode too short: {n} < {self.config.min_episode_frames}")
        if not n:
            raise CaptureError("empty episode")
        numeric = {
            "observation.state": np.stack([r[1] for r in records]).astype(np.float32),
            "action": np.stack([r[2] for r in records]).astype(np.float32),
            "timestamp": (np.arange(n) / self.config.fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
        }
        expected_times = records[0][0] + np.arange(n) / self.config.fps
        if not np.allclose([r[0] for r in records], expected_times, rtol=0, atol=1e-5):
            issues.append("capture timestamps do not follow the fixed sampling grid")
        table = pa.table(
            {
                key: pa.array(value.tolist(), type=pa.list_(pa.float32(), len(self.config.names)))
                if value.ndim == 2
                else pa.array(value)
                for key, value in numeric.items()
            }
        )
        pq.write_table(table, self.directory / "frames.parquet", compression="zstd")
        stats = {key: numeric_stats(value) for key, value in numeric.items()}
        for key, m in moments.items():
            mean = m["sum"] / m["pixels"]
            std = np.sqrt(np.maximum(0, m["sumsq"] / m["pixels"] - mean**2))
            stats[key] = {
                k: v.reshape(3, 1, 1).tolist()
                for k, v in {"min": m["min"], "max": m["max"], "mean": mean, "std": std}.items()
            }
            stats[key]["count"] = [n]
        repeats = {}
        for name, values in sequences.items():
            repeated = sum(a == b for a, b in zip(values, values[1:]))
            repeats[name] = repeated
            if repeated / max(1, n - 1) > 0.10:
                issues.append(f"{name}: more than 10% repeated source frames")
        write_json(
            self.directory / "manifest.json",
            {
                "task": self.task,
                "length": n,
                "mock": self.mock,
                "contract": self.config.contract(),
                "stats": stats,
                "issues": issues,
                "camera_repeat_counts": repeats,
                "capture_start": records[0][0],
                "capture_end": records[-1][0],
            },
        )


class DatasetStore(AbstractContextManager):
    """Single writer, recoverable commit journal, standard LeRobot v2.1 public files."""

    def __init__(self, config: CaptureConfig, *, resume: bool = False, mock: bool = False):
        self.config, self.mock = config, mock
        self.root = Path(config.output).expanduser().resolve()
        if self.root.exists() and not resume:
            raise FileExistsError(f"{self.root} exists; use --resume for a piper-capture dataset")
        if resume and not (self.root / ".piper/contract.json").exists():
            raise ValueError("resume requires a dataset created by piper-capture")
        self.root.mkdir(parents=True, exist_ok=True)
        self.internal = self.root / ".piper"
        self.internal.mkdir(exist_ok=True)
        self._lock = (self.internal / "writer.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise CaptureError("another process is writing this dataset") from None
        try:
            self.contract = {**config.contract(), "mock": mock}
            contract_path = self.internal / "contract.json"
            if contract_path.exists():
                if read_json(contract_path) != self.contract:
                    raise ValueError(
                        "dataset contract differs (fps, cameras, arms, units, or simulation mode)"
                    )
            else:
                write_json(contract_path, self.contract)
            (self.internal / "committed").mkdir(exist_ok=True)
            self._recover_delete()
            self._recover_commit()
            self._write_metadata()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def records(self) -> list[dict]:
        return [read_json(p) for p in sorted((self.internal / "committed").glob("*.json"))]

    def start(self, task: str) -> EpisodeWriter:
        if not task.strip():
            raise ValueError("task cannot be empty")
        if shutil.disk_usage(self.root).free < self.config.min_free_gb * 1e9:
            raise CaptureError(f"less than {self.config.min_free_gb} GB disk space remaining")
        return EpisodeWriter(self.internal / "pending" / uuid.uuid4().hex, self.config, task, self.mock)

    def pending(self) -> list[dict]:
        return [
            {
                "id": p.name,
                **(
                    read_json(p / "manifest.json")
                    if (p / "manifest.json").exists()
                    else {"issues": ["incomplete recording"], "length": None}
                ),
            }
            for p in sorted((self.internal / "pending").glob("*"))
            if p.is_dir()
        ]

    def pending_path(self, identifier: str) -> Path:
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("invalid pending recording ID")
        path = self.internal / "pending" / identifier
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    def reject(self, directory: Path, reason: str = "operator discarded") -> Path:
        directory = self.pending_path(directory.name)
        target = self.internal / "rejected" / directory.name
        target.parent.mkdir(exist_ok=True)
        write_json(directory / "rejection.json", {"reason": reason})
        os.replace(directory, target)
        return target

    def failed_records(self) -> list[dict]:
        return [
            {**read_json(p), "id": p.parent.name, "outcome": "failure"}
            for p in sorted((self.internal / "failed").glob("*/manifest.json"))
        ]

    def save_failure(self, directory: Path) -> str:
        directory = self.pending_path(directory.name)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            capture = read_json(directory / "capture.json")
            manifest = {"task": capture["task"], "length": 0, "issues": ["incomplete recording"]}
        write_json(manifest_path, {**manifest, "outcome": "failure"})
        target = self.internal / "failed" / directory.name
        target.parent.mkdir(exist_ok=True)
        os.replace(directory, target)
        return target.name

    def _selection(self, selections: list[dict]) -> tuple[list[dict], list[dict]]:
        saved = {str(r["episode_index"]): r for r in self.records()}
        failed = {r["id"]: r for r in self.failed_records()}
        chosen: dict[str, list] = {"saved": [], "failed": []}
        seen = set()
        if not selections:
            raise ValueError("请选择至少一条轨迹")
        for item in selections:
            kind, identifier = item["kind"], item["id"]
            if (kind, identifier) in seen:
                raise ValueError("轨迹重复选择")
            seen.add((kind, identifier))
            source = saved if kind == "saved" else failed if kind == "failed" else {}
            if identifier not in source:
                raise ValueError("所选轨迹不存在，请刷新记录")
            chosen[kind].append(source[identifier])
        return chosen["saved"], chosen["failed"]

    def _subset(self, destination: Path, records: list[dict], *, copy_files: bool = False):
        """Build a contiguous LeRobot dataset; external exports own their files."""
        transfer = shutil.copy2 if copy_files else os.link
        with DatasetStore(replace(self.config, output=str(destination)), mock=self.mock) as subset:
            tasks: list[str] = []
            offset = 0
            for index, record in enumerate(sorted(records, key=lambda r: r["episode_index"])):
                record = {**record, "stats": dict(record["stats"])}
                old = {
                    "episode_index": record["episode_index"],
                    "episode_chunk": record["episode_index"] // 1000,
                }
                new = {"episode_index": index, "episode_chunk": index // 1000}
                if record["task"] not in tasks:
                    tasks.append(record["task"])
                task_index, n = tasks.index(record["task"]), record["length"]
                table = pq.read_table(self.root / DATA_PATH.format(**old))
                for key, values in {
                    "episode_index": np.full(n, index, dtype=np.int64),
                    "index": np.arange(offset, offset + n, dtype=np.int64),
                    "task_index": np.full(n, task_index, dtype=np.int64),
                }.items():
                    table = table.set_column(table.schema.get_field_index(key), key, pa.array(values))
                    record["stats"][key] = numeric_stats(values)
                data = destination / DATA_PATH.format(**new)
                data.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, data, compression="zstd")
                for camera in self.config.cameras:
                    source = self.root / VIDEO_PATH.format(video_key=camera.key, **old)
                    target = destination / VIDEO_PATH.format(video_key=camera.key, **new)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    transfer(source, target)
                for suffix in ("timing.jsonl", "config.json"):
                    source = self.root / f"meta/capture/episode_{old['episode_index']:06d}.{suffix}"
                    target = destination / f"meta/capture/episode_{index:06d}.{suffix}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    transfer(source, target)
                record.update(episode_index=index, task_index=task_index)
                write_json(subset.internal / "committed" / f"{index:06d}.json", record)
                offset += n
            subset._write_metadata()

    def delete_selected(self, selections: list[dict]):
        saved, failed = self._selection(selections)
        removed = {r["episode_index"] for r in saved}
        stage = self.internal / f"delete-{uuid.uuid4().hex}"
        stage.mkdir()
        try:
            swaps = []
            if saved:
                self._subset(
                    stage / "replacement", [r for r in self.records() if r["episode_index"] not in removed]
                )
                for part in ("data", "videos", "meta", ".piper/committed"):
                    (stage / "replacement" / part).mkdir(parents=True, exist_ok=True)
                    swaps.append(part)
            write_json(
                self.internal / "delete.json",
                {"stage": stage.name, "swaps": swaps, "failed": [r["id"] for r in failed]},
            )
        except BaseException:
            shutil.rmtree(stage)
            raise
        self._recover_delete()

    def _recover_delete(self):
        journal = self.internal / "delete.json"
        if not journal.exists():
            return
        job = read_json(journal)
        stage = self.internal / job["stage"]
        # Each rename is replayable: the staged replacement disappears only after installation.
        for part in job["swaps"]:
            replacement, current, backup = (
                stage / "replacement" / part,
                self.root / part,
                stage / "old" / part,
            )
            if replacement.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                if current.exists() and not backup.exists():
                    os.replace(current, backup)
                os.replace(replacement, current)
        for identifier in job["failed"]:
            directory = self.internal / "failed" / identifier
            if directory.exists():
                shutil.rmtree(directory)
        journal.unlink()
        shutil.rmtree(stage)

    def export_selected(self, selections: list[dict], destination: str | None = None) -> Path:
        saved, failed = self._selection(selections)
        target = None
        if destination is not None:
            target = Path(destination.strip()).expanduser()
            if not target.is_absolute():
                raise ValueError("导出目录必须是本机绝对路径")
            target = target.resolve()
            if target == self.root or self.root in target.parents or target in self.root.parents:
                raise ValueError("导出目录不能与当前采集目录重叠")
            if target.exists():
                raise ValueError("导出目录已存在，请填写新的目录名")
            target.parent.mkdir(parents=True, exist_ok=True)
        directory = (
            target.parent / f".piper-export-{uuid.uuid4().hex}"
            if target
            else self.internal / "exports" / uuid.uuid4().hex
        )
        directory.mkdir(parents=True)
        try:
            mapping = []
            if saved:
                self._subset(directory / "success", saved, copy_files=target is not None)
                for index, record in enumerate(sorted(saved, key=lambda r: r["episode_index"])):
                    mapping.append(
                        {
                            "kind": "saved",
                            "source_id": str(record["episode_index"]),
                            "recording_id": record.get("recording_id"),
                            "export_name": f"success/episode_{index:06d}",
                        }
                    )
            if failed:
                for index, record in enumerate(failed):
                    name = f"episode_{index:06d}"
                    episode = directory / "failed" / name
                    shutil.copytree(
                        self.internal / "failed" / record["id"],
                        episode,
                        copy_function=shutil.copy2 if target else os.link,
                    )
                    if (episode / "frames.parquet").exists():
                        (episode / "frames.parquet").rename(episode / f"{name}.parquet")
                    for camera in self.config.cameras:
                        video = episode / f"{camera.key}.mp4"
                        if video.exists():
                            video.parent.joinpath(camera.key).mkdir()
                            video.rename(episode / camera.key / f"{name}.mp4")
                    mapping.append(
                        {"kind": "failed", "source_id": record["id"], "export_name": f"failed/{name}"}
                    )
            write_json(
                directory / "selection.json",
                {
                    "format": "piper_capture_export_v2",
                    "selections": selections,
                    "success_count": len(saved),
                    "failure_count": len(failed),
                    "episodes": mapping,
                },
            )
            if target:
                # The completed tree appears only after all data and metadata have been written.
                # Refuse an existing destination, including one created during this export.
                target.mkdir(exist_ok=False)
                try:
                    os.replace(directory, target)
                except BaseException:
                    target.rmdir()
                    raise
                return target
            archive = directory / "recordings.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
                for path in sorted(directory.rglob("*")):
                    if (
                        path.is_file()
                        and path != archive
                        and ".piper" not in path.relative_to(directory).parts
                    ):
                        output.write(path, path.relative_to(directory))
            for part in ("success", "failed"):
                if (directory / part).exists():
                    shutil.rmtree(directory / part)
            return archive
        except BaseException:
            shutil.rmtree(directory)
            raise

    def accept(self, directory: Path) -> int:
        from .validate import validate_pending

        directory = self.pending_path(directory.name)
        manifest = read_json(directory / "manifest.json")
        if manifest["issues"]:
            raise CaptureError(f"episode failed quality checks: {manifest['issues']}")
        if manifest["contract"] != self.config.contract() or manifest["mock"] != self.mock:
            raise ValueError("pending episode contract does not match dataset")
        validate_pending(directory)
        records = self.records()
        index, total_frames = len(records), sum(r["length"] for r in records)
        tasks = list(dict.fromkeys(r["task"] for r in records))
        if manifest["task"] not in tasks:
            tasks.append(manifest["task"])
        task_index = tasks.index(manifest["task"])
        n = manifest["length"]
        table = pq.read_table(directory / "frames.parquet")
        for key, values in {
            "episode_index": np.full(n, index, dtype=np.int64),
            "index": np.arange(total_frames, total_frames + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
        }.items():
            table = table.append_column(key, pa.array(values))
            manifest["stats"][key] = numeric_stats(values)
        pq.write_table(table, directory / "export.parquet", compression="zstd")
        paths = {"export.parquet": DATA_PATH.format(episode_chunk=index // 1000, episode_index=index)}
        for camera in self.config.cameras:
            paths[f"{camera.key}.mp4"] = VIDEO_PATH.format(
                episode_chunk=index // 1000, episode_index=index, video_key=camera.key
            )
        paths["timing.jsonl"] = f"meta/capture/episode_{index:06d}.timing.jsonl"
        paths["capture.json"] = f"meta/capture/episode_{index:06d}.config.json"
        record = {
            **manifest,
            "recording_id": directory.name,
            "outcome": "success",
            "episode_index": index,
            "task_index": task_index,
        }
        # Once this journal is durable, resume completes the commit idempotently.
        write_json(
            self.internal / "commit.json", {"pending": directory.name, "paths": paths, "record": record}
        )
        self._recover_commit()
        self._write_metadata()
        return index

    def _recover_commit(self):
        path = self.internal / "commit.json"
        if not path.exists():
            return
        job = read_json(path)
        source = self.internal / "pending" / job["pending"]
        for src, dst in job["paths"].items():
            origin, destination = source / src, self.root / dst
            destination.parent.mkdir(parents=True, exist_ok=True)
            if origin.exists():
                if destination.exists():
                    raise CaptureError(f"commit would overwrite an existing file: {destination}")
                os.replace(origin, destination)
            elif not destination.is_file():
                raise CaptureError(f"interrupted commit is missing {src}")
        index = job["record"]["episode_index"]
        write_json(self.internal / "committed" / f"{index:06d}.json", job["record"])
        self._write_metadata()
        if source.exists():
            shutil.rmtree(source)
        path.unlink()

    def _write_metadata(self):
        records = self.records()
        tasks = list(dict.fromkeys(r["task"] for r in records))
        n, total_frames = len(records), sum(r["length"] for r in records)
        features = {
            key: {"dtype": "float32", "shape": [len(self.config.names)], "names": self.config.names}
            for key in ("observation.state", "action")
        }
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            features[key] = {
                "dtype": "float32" if key == "timestamp" else "int64",
                "shape": [1],
                "names": None,
            }
        for c in self.config.cameras:
            features[c.key] = {
                "dtype": "video",
                "shape": [c.height, c.width, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": self.config.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.height": c.height,
                    "video.width": c.width,
                    "video.channels": 3,
                    "has_audio": False,
                    "is_depth_map": False,
                },
            }
        meta = self.root / "meta"
        write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(tasks)])
        write_jsonl(
            meta / "episodes.jsonl",
            [
                {
                    "episode_index": r["episode_index"],
                    "tasks": [r["task"]],
                    "length": r["length"],
                    "quality": "green",
                }
                for r in records
            ],
        )
        write_jsonl(
            meta / "episodes_stats.jsonl",
            [{"episode_index": r["episode_index"], "stats": r["stats"]} for r in records],
        )
        write_json(meta / "stats.json", aggregate_stats([r["stats"] for r in records]))
        write_json(meta / "capture_contract.json", self.contract)
        write_json(
            meta / "info.json",
            {
                "codebase_version": "v2.1",
                "robot_type": "agilex",
                "fps": self.config.fps,
                "total_episodes": n,
                "total_frames": total_frames,
                "total_tasks": len(tasks),
                "total_videos": n * len(self.config.cameras),
                "total_chunks": math.ceil(n / 1000),
                "chunks_size": 1000,
                "splits": {"train": f"0:{n}"},
                "data_path": DATA_PATH,
                "video_path": VIDEO_PATH,
                "features": features,
            },
        )
