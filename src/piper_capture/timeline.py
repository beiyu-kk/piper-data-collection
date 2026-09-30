from __future__ import annotations

import threading
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import CaptureConfig


class CaptureError(RuntimeError):
    """Incomplete, stale, or invalid source data must not enter training data."""


@dataclass(frozen=True)
class Sample:
    time: float
    value: np.ndarray
    sequence: int = -1
    device_time_ms: float = -1.0


@dataclass(frozen=True)
class Frame:
    time: float
    state: np.ndarray
    action: np.ndarray
    images: dict[str, np.ndarray]
    timing: dict


class Timeline:
    def __init__(self, maxlen: int):
        self._samples: deque[Sample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: Sample) -> None:
        with self._lock:
            if self._samples and sample.time <= self._samples[-1].time:
                raise CaptureError("source timestamp did not increase")
            self._samples.append(sample)

    def samples(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)

    def latest(self) -> Sample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def vector(self, timestamp: float, max_gap: float, *, hold: bool = False) -> tuple[np.ndarray, list]:
        samples = self.samples()
        times = [s.time for s in samples]
        if hold:
            index = bisect_right(times, timestamp) - 1
            if index < 0 or timestamp - times[index] > max_gap:
                raise CaptureError("missing or stale gripper sample")
            return samples[index].value.copy(), [times[index], times[index]]
        index = bisect_left(times, timestamp)
        if index < len(times) and times[index] == timestamp:
            return samples[index].value.copy(), [timestamp, timestamp]
        if index == 0 or index == len(times):
            raise CaptureError("joint stream does not bracket the requested timestamp")
        left, right = samples[index - 1], samples[index]
        if right.time - left.time > max_gap:
            raise CaptureError("joint stream has a timestamp gap")
        alpha = (timestamp - left.time) / (right.time - left.time)
        return (left.value + alpha * (right.value - left.value)).astype(np.float32), [left.time, right.time]

    def image(self, timestamp: float, max_skew: float) -> Sample:
        samples = self.samples()
        if not samples:
            raise CaptureError("camera has no frames")
        times = [s.time for s in samples]
        index = bisect_left(times, timestamp)
        candidates = samples[max(0, index - 1) : min(len(samples), index + 1)]
        sample = min(candidates, key=lambda s: abs(s.time - timestamp))
        if abs(sample.time - timestamp) > max_skew:
            raise CaptureError("camera timestamp skew exceeds limit")
        return sample


class Aligner:
    def __init__(self, config: CaptureConfig, vectors: dict[str, Timeline], cameras: dict[str, Timeline]):
        self.config = config
        self.vectors = vectors
        self.cameras = cameras

    def frame(self, timestamp: float, *, require_action: bool = True) -> Frame:
        arrays = {}
        timing: dict = {"vectors": {}, "cameras": {}}
        for field in ("state", "action") if require_action else ("state",):
            parts = []
            for arm in self.config.arms:
                for group in ("12", "34", "56", "gripper"):
                    key = f"{arm.name}.{field}.{group}"
                    try:
                        value, bracket = self.vectors[key].vector(
                            timestamp,
                            float("inf") if field == "action" else self.config.max_vector_gap_s,
                            hold=field == "action" or group == "gripper",
                        )
                    except CaptureError as exc:
                        raise CaptureError(f"{key}: {exc}") from exc
                    parts.append(value)
                    timing["vectors"][key] = bracket
            arrays[field] = np.concatenate(parts).astype(np.float32)
            if arrays[field].shape != (len(self.config.names),) or not np.isfinite(arrays[field]).all():
                raise CaptureError(f"invalid {field} vector")
        images = {}
        for camera in self.config.cameras:
            try:
                sample = self.cameras[camera.name].image(timestamp, self.config.max_camera_skew_s)
            except CaptureError as exc:
                raise CaptureError(f"{camera.name}: {exc}") from exc
            if sample.value.shape != (camera.height, camera.width, 3) or sample.value.dtype != np.uint8:
                raise CaptureError(f"{camera.name}: unexpected RGB frame shape or dtype")
            images[camera.key] = sample.value
            timing["cameras"][camera.name] = {
                "host_time": sample.time,
                "sequence": sample.sequence,
                "device_time_ms": sample.device_time_ms,
                "skew_s": sample.time - timestamp,
            }
        return Frame(
            timestamp, arrays["state"], arrays.get("action", np.empty(0, np.float32)), images, timing
        )
