from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraConfig:
    name: str
    backend: str = "realsense"
    serial: str = ""
    device: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30

    @property
    def key(self) -> str:
        return f"observation.images.{self.name}"


@dataclass(frozen=True)
class ArmConfig:
    name: str = "right"
    state_can: str = "can0"
    command_can: str = "can0"


@dataclass(frozen=True)
class CaptureConfig:
    output: str
    task: str
    fps: int = 30
    arms: tuple[ArmConfig, ...] = field(default_factory=lambda: (ArmConfig(),))
    cameras: tuple[CameraConfig, ...] = ()
    alignment_delay_s: float = 0.12
    max_camera_skew_s: float = 0.05
    max_vector_gap_s: float = 0.08
    history_s: float = 2.0
    min_episode_frames: int = 30
    max_episode_s: float = 180.0
    queue_size: int = 16
    min_free_gb: float = 5.0

    def validate(self) -> CaptureConfig:
        if not self.output.strip() or not self.task.strip():
            raise ValueError("output and task must not be empty")
        for name in ("fps", "min_episode_frames", "queue_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.arms or not self.cameras:
            raise ValueError("at least one arm and camera are required")
        if [a.name for a in self.arms] not in (["right"], ["left", "right"]):
            raise ValueError("arms must be [right] or [left, right], in that order")
        for channel in ("state_can", "command_can"):
            ports = [getattr(a, channel) for a in self.arms]
            if len(set(ports)) != len(ports) or not all(ports):
                raise ValueError(f"each arm needs a distinct nonempty {channel}")
        if len({c.name for c in self.cameras}) != len(self.cameras):
            raise ValueError("camera names must be unique")
        sources = []
        for c in self.cameras:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", c.name):
                raise ValueError(f"invalid camera name: {c.name}")
            if c.backend not in {"realsense", "opencv"}:
                raise ValueError(f"invalid camera backend: {c.backend}")
            if any(type(value) is not int for value in (c.width, c.height, c.fps)):
                raise ValueError("camera width, height and fps must be integers")
            if min(c.width, c.height, c.fps) <= 0 or c.width % 2 or c.height % 2:
                raise ValueError("video dimensions must be positive and even; fps must be positive")
            if c.fps < self.fps:
                raise ValueError("camera fps must be at least collection fps")
            if c.backend == "realsense" and not c.serial:
                raise ValueError(f"RealSense serial is required for {c.name}")
            sources.append((c.backend, str(c.serial if c.backend == "realsense" else c.device)))
        if len(set(sources)) != len(sources):
            raise ValueError("the same physical camera cannot be opened twice")
        for name in (
            "alignment_delay_s",
            "max_camera_skew_s",
            "max_vector_gap_s",
            "history_s",
            "max_episode_s",
            "min_free_gb",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.alignment_delay_s <= max(self.max_camera_skew_s, self.max_vector_gap_s):
            raise ValueError("alignment_delay_s must exceed camera skew and vector gap limits")
        if self.history_s <= self.alignment_delay_s * 2:
            raise ValueError("history_s must exceed twice alignment_delay_s")
        if self.min_episode_frames < 2 or self.queue_size < 1:
            raise ValueError("min_episode_frames >= 2 and queue_size >= 1 are required")
        if self.min_episode_frames > self.max_episode_s * self.fps:
            raise ValueError("minimum episode length exceeds maximum")
        return self

    @property
    def names(self) -> list[str]:
        return [f"{a.name}_joint_{i}" for a in self.arms for i in range(7)]

    def as_dict(self) -> dict:
        return asdict(self)

    def contract(self) -> dict:
        return {
            "format": "piper_capture_v1",
            "fps": self.fps,
            "names": self.names,
            "cameras": {c.key: [c.height, c.width, 3] for c in self.cameras},
            "action_source": "leader_command",
            "action_representation": "absolute",
            "joint_unit": "radian",
            "gripper_unit": "meter",
            "gripper_representation": "SDK total opening",
            "alignment": "linear joints; previous gripper; nearest RGB; host receive clock",
            "openpi_action_delta_timestamps_start": 0,
        }


def load_config(path: str | Path) -> CaptureConfig:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping")
    try:
        data["arms"] = tuple(ArmConfig(**a) for a in data.get("arms", [{}]))
        data["cameras"] = tuple(
            CameraConfig(**{**c, "serial": str(c.get("serial", ""))}) for c in data.get("cameras", [])
        )
        return CaptureConfig(**data).validate()
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"invalid capture configuration: {exc}") from exc
