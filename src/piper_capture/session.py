from __future__ import annotations

import shutil
import time

from .sources import Sources
from .storage import DatasetStore, EpisodeWriter
from .timeline import CaptureError


class CaptureSession:
    def __init__(self, sources: Sources, store: DatasetStore):
        self.sources, self.store = sources, store
        self.config = store.config
        self.writer: EpisodeWriter | None = None
        self.start_time = 0.0
        self.last_disk_check = 0.0
        self.task = self.config.task
        self.awaiting_action = False

    def start(self, *, allow_wait: bool = False):
        if self.writer is not None:
            raise CaptureError("an episode is already recording")
        self.sources.wait_ready(require_action=not allow_wait)
        self.awaiting_action = allow_wait and any(
            stream.latest() is None for key, stream in self.sources.vectors.items() if ".action." in key
        )
        self.writer = self.store.start(self.task)
        # Start at the operator's command time, then wait for the alignment look-ahead.
        self.start_time = time.monotonic()

    def tick(self) -> bool:
        if self.writer is None:
            return False
        self.sources.check()
        now = time.monotonic()
        if self.awaiting_action:
            # The operator may arm capture while the event-driven leader is still silent.
            self.sources.aligner.frame(now - self.config.alignment_delay_s, require_action=False)
            if any(
                stream.latest() is None for key, stream in self.sources.vectors.items() if ".action." in key
            ):
                if now - self.start_time >= self.config.max_episode_s:
                    raise CaptureError("等待首组主臂目标超时，未生成有效示教帧")
                return False
            self.awaiting_action = False
            self.start_time = now
        if now - self.last_disk_check > 1:
            if shutil.disk_usage(self.store.root).free < self.config.min_free_gb * 1e9:
                raise CaptureError("disk free space fell below the configured minimum")
            self.last_disk_check = now
        timestamp = self.start_time + self.writer.count / self.config.fps
        if now < timestamp + self.config.alignment_delay_s:
            return False
        if now - timestamp > self.config.history_s / 2:
            raise CaptureError("capture loop fell behind its fixed sampling grid")
        frame = self.sources.aligner.frame(timestamp)
        self.writer.append(frame)
        return self.writer.count >= int(self.config.max_episode_s * self.config.fps)

    def stop(self, failure: str | None = None):
        if self.writer is None:
            raise CaptureError("no recording is active")
        writer, self.writer = self.writer, None
        self.awaiting_action = False
        return writer.finish(failure)
