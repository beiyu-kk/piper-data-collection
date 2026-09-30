from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import queue
import secrets
import shutil
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import CaptureConfig
from .replay import ReplayPlan, RobotReplay
from .session import CaptureSession
from .sources import Sources
from .storage import DATA_PATH, VIDEO_PATH, DatasetStore, read_json
from .timeline import CaptureError

LOGGER = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Selection(BaseModel):
    kind: Literal["saved", "failed"]
    id: str = Field(max_length=64)


class Command(BaseModel):
    command: str
    task: str | None = Field(default=None, max_length=2000)
    recording_id: str | None = None
    selections: list[Selection] = Field(default_factory=list, max_length=5000)
    revision: str | None = None
    recording_ids: list[str] = Field(default_factory=list, max_length=5000)
    decision: Literal["accept", "mark_failure", "discard"] | None = None
    destination: str | None = Field(default=None, max_length=4096)
    kind: Literal["saved", "failed", "pending"] = "saved"
    field: Literal["observation.state", "action"] = "action"
    with_gripper: bool = False
    speed: Literal[0.25, 0.5, 1.0] = 1.0
    plan_token: str | None = None
    replay_id: str | None = None
    leader_disconnected: bool = False
    workspace_clear: bool = False
    recording_revision: str | None = None


class WebRuntime:
    """Only this worker owns the session and mutates the dataset; HTTP never ticks capture."""

    def __init__(self, config: CaptureConfig, *, mock: bool = False, resume: bool = False):
        self.config, self.mock, self.resume = config, mock, resume
        self.commands: queue.Queue[tuple[Command, Future]] = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.catalog_lock = threading.RLock()
        self.snapshot: dict = {"state": "CONNECTING", "error": "", "ready": False}
        self.jpeg: dict[str, tuple[float, bytes]] = {}
        self.sources: Sources | None = None
        self.store: DatasetStore | None = None
        self.session: CaptureSession | None = None
        self.state = "CONNECTING"
        self.error = ""
        self.pending_rows: list[dict] = []
        self.history_rows: list[dict] = []
        self.events: list[dict] = []
        self.instance = secrets.token_hex(16)
        self.sequence = 0
        self.replay: RobotReplay | None = None
        self.replay_plan: ReplayPlan | None = None
        self.replay_needs_reconnect = False
        self.thread = threading.Thread(target=self._run, name="web-capture", daemon=True)
        self.preview_thread = threading.Thread(target=self._preview, name="web-preview", daemon=True)

    def start(self):
        self.thread.start()
        self.preview_thread.start()

    def close(self):
        self.stop_event.set()
        if self.replay:
            self.replay.close()
        self.thread.join()
        self.preview_thread.join()

    def submit(self, command: Command) -> Future:
        future: Future = Future()
        if command.command in {"replay_stop", "replay_heartbeat"}:
            if self.replay and command.replay_id == self.replay.id:
                if command.command == "replay_stop":
                    self.replay.cancel.set()
                else:
                    self.replay.heartbeat()
                future.set_result({"ok": True})
            else:
                future.set_exception(CaptureError("回放会话已失效"))
            return future
        if not self.thread.is_alive() or self.stop_event.is_set():
            future.set_exception(CaptureError("采集服务未运行"))
            return future
        try:
            self.commands.put_nowait((command, future))
        except queue.Full:
            future.set_exception(CaptureError("操作队列已满，请等待当前操作完成"))
        return future

    def status(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.snapshot)

    def event(self, message: str, kind: str = "info"):
        self.events.append({"time": time.time(), "message": message, "kind": kind})
        self.events = self.events[-30:]

    def _catalog(self):
        self.pending_rows = [
            {k: row.get(k) for k in ("id", "task", "length", "issues", "capture_start", "capture_end")}
            for row in self.store.pending()
        ]
        self.history_rows = [
            {
                "id": str(r["episode_index"]),
                "episode_index": r["episode_index"],
                "task": r["task"],
                "length": r["length"],
                "issues": [],
                "kind": "saved",
                "outcome": "success",
                "recording_id": r.get("recording_id", str(r.get("capture_start"))),
            }
            for r in reversed(self.store.records())
        ]
        self.history_rows.extend(
            {k: r.get(k) for k in ("id", "task", "length", "issues", "outcome")} | {"kind": "failed"}
            for r in reversed(self.store.failed_records())
        )

    def catalog_revision(self):
        return hashlib.sha256(json.dumps(self.history_rows, sort_keys=True).encode()).hexdigest()

    def _connect(self):
        if self.sources is not None:
            self.sources.__exit__(None, None, None)
        self.sources = Sources(self.config, mock=self.mock)
        self.sources.__enter__()
        task = self.session.task if self.session else self.config.task
        self.session = CaptureSession(self.sources, self.store)
        self.session.task = task
        self.error = ""
        self.state = "IDLE"
        self.event("模拟设备已启动" if self.mock else "设备监听已启动")

    def _run(self):
        try:
            self.store = DatasetStore(self.config, resume=self.resume, mock=self.mock)
            self._catalog()
            self._connect()
            next_status = 0.0
            while not self.stop_event.is_set():
                try:
                    if self.session.writer and self.session.tick():
                        self._finish()
                except Exception as exc:
                    LOGGER.exception("Capture failed")
                    self._finish(str(exc))
                try:
                    command, future = self.commands.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if future.set_running_or_notify_cancel():
                        try:
                            with self.catalog_lock:
                                result = self._execute(command)
                            self._publish()
                            future.set_result({"ok": True, **(result or {}), "status": self.status()})
                        except Exception as exc:
                            LOGGER.warning("Command %s failed: %s", command.command, exc)
                            self.error = str(exc)
                            if self.state == "SAVING":
                                self.state = "IDLE"
                            self._catalog()
                            self.event(str(exc), "error")
                            self._publish()
                            future.set_exception(CaptureError(str(exc)))
                if time.monotonic() >= next_status:
                    self._publish()
                    next_status = time.monotonic() + 0.2
                self.stop_event.wait(0.002)
        except Exception as exc:
            LOGGER.exception("Web capture runtime failed")
            self.error, self.state = str(exc), "ERROR"
            self._publish()
        finally:
            if self.replay:
                self.replay.close()
            if self.session and self.session.writer:
                self._finish("服务关闭，录制中断")
            if self.sources:
                self.sources.__exit__(None, None, None)
            if self.store:
                self.store.close()
            while not self.commands.empty():
                _, future = self.commands.get_nowait()
                if not future.done():
                    future.set_exception(CaptureError("采集服务已停止"))

    def _finish(self, failure: str | None = None):
        self.state = "SAVING"
        self._publish()
        try:
            directory = self.session.stop(failure=failure)
            self.event(
                "录制异常，数据已保留" if failure else "录制结束，等待审核", "error" if failure else "info"
            )
            self.error = failure or ""
            return directory
        except Exception as exc:
            self.error = str(exc)
            self.event(str(exc), "error")
        finally:
            self.state = "IDLE"
            self._catalog()

    def _execute(self, command: Command):
        if self.replay and self.replay.status()["active"]:
            raise CaptureError("真机回放期间不能采集或修改数据，请先停止回放")
        if (self.store.internal / "delete.json").exists() or (self.store.internal / "commit.json").exists():
            self.store._recover_delete()
            self.store._recover_commit()
            self._catalog()
        if command.command == "start":
            if self.replay_needs_reconnect:
                raise CaptureError("回放已结束，请恢复硬件主从连接并重新连接采集设备")
            if self.session.writer:
                raise CaptureError("已有轨迹正在录制")
            if command.task is None or not command.task.strip():
                raise CaptureError("请填写任务指令")
            self.sources.check()
            self.sources.aligner.frame(time.monotonic() - self.config.alignment_delay_s, require_action=False)
            self.session.task = command.task.strip()
            self.session.start(allow_wait=True)
            self.state, self.error = "RECORDING", ""
            self.event("已开始，等待首组主臂目标" if self.session.awaiting_action else "开始录制")
        elif command.command in {"stop", "save_success", "save_failure"}:
            if not self.session.writer:
                raise CaptureError("当前没有正在录制的轨迹")
            pending = self.session.writer.directory
            directory = self._finish()
            if command.command == "save_failure":
                self.store.save_failure(directory or pending)
                self.event("失败示教已保存", "info")
                self.error = ""
                self._catalog()
            elif directory is None:
                raise CaptureError(self.error)
            elif command.command == "save_success":
                self.state = "SAVING"
                self._publish()
                index = self.store.accept(directory)
                self.event(f"成功示教 {index:06d} 已保存", "success")
                self.error, self.state = "", "IDLE"
                self._catalog()
        elif command.command in {"accept", "discard", "mark_failure"}:
            if self.session.writer:
                raise CaptureError("请先停止录制")
            if not command.recording_id:
                raise CaptureError("缺少轨迹 ID")
            path = self.store.pending_path(command.recording_id)
            self.state = "SAVING"
            self._publish()
            if command.command == "accept":
                index = self.store.accept(path)
                self.event(f"轨迹 {index:06d} 已保存", "success")
            elif command.command == "mark_failure":
                self.store.save_failure(path)
                self.event("失败示教已保存")
            else:
                self.store.reject(path)
                self.event("轨迹已丢弃，诊断文件保留")
            self.error, self.state = "", "IDLE"
            self._catalog()
        elif command.command == "review_batch":
            if self.session.writer:
                raise CaptureError("请先停止录制")
            ids = command.recording_ids
            if not ids or len(set(ids)) != len(ids) or command.decision is None:
                raise CaptureError("请选择轨迹和审核结果，轨迹不能重复")
            paths = [self.store.pending_path(identifier) for identifier in ids]
            self.state = "SAVING"
            self._publish()
            results = []
            for path in paths:
                try:
                    if command.decision == "accept":
                        index = self.store.accept(path)
                        result = {"episode_index": index}
                    elif command.decision == "mark_failure":
                        self.store.save_failure(path)
                        result = {}
                    else:
                        self.store.reject(path)
                        result = {}
                    results.append({"id": path.name, "ok": True, **result})
                except Exception as exc:
                    results.append({"id": path.name, "ok": False, "error": str(exc)})
                    if (self.store.internal / "commit.json").exists():
                        break
            completed = sum(row["ok"] for row in results)
            self.error, self.state = "", "IDLE"
            self._catalog()
            self.event(f"批量审核完成：{completed} / {len(ids)} 条已处理")
            return {"results": results, "processed": completed, "remaining": len(ids) - completed}
        elif command.command == "reconnect":
            if self.session.writer:
                raise CaptureError("录制期间不能重新连接设备")
            self._connect()
            self.replay_needs_reconnect = False
            self.replay_plan = None
        elif command.command == "replay_prepare":
            self.replay_plan = None
            if self.session.writer:
                raise CaptureError("请先停止录制")
            metadata, data, _ = self.recording(command.kind, command.recording_id or "")
            if command.recording_revision != recording_revision(metadata):
                raise CaptureError("轨迹已更新，请关闭窗口并重新打开轨迹")
            if metadata.get("mock") != self.mock:
                raise CaptureError("不能将模拟数据用于真机回放")
            self.replay_plan = ReplayPlan.load(
                data, metadata, self.config, command.field, command.speed, command.with_gripper
            )
            return {"plan": self.replay_plan.describe()}
        elif command.command == "replay_start":
            if self.session.writer:
                raise CaptureError("请先停止录制")
            plan = self.replay_plan
            if plan is None or command.plan_token != plan.token or time.monotonic() - plan.created > 120:
                raise CaptureError("回放预检已过期，请重新预检")
            if not command.leader_disconnected or not command.workspace_clear:
                raise CaptureError("请确认主臂已断开控制连接、从臂已使能，回起点及回放路径无障碍")
            self.replay = RobotReplay(plan, self.config.arms[0].state_can, mock=self.mock)
            self.replay_plan = None
            self.replay_needs_reconnect = True
            self.replay.thread.start()
            self.event("模拟回放开始" if self.mock else "真机回放预检开始")
            return {"replay_id": self.replay.id}
        elif command.command in {"delete", "export"}:
            if self.session.writer:
                raise CaptureError("请先结束当前录制")
            if command.revision != self.catalog_revision():
                raise CaptureError("记录已更新，请重新选择轨迹")
            self.state = "SAVING"
            self._publish()
            selections = [item.model_dump() for item in command.selections]
            if command.command == "delete":
                self.store.delete_selected(selections)
                self.replay_plan = None
                self._catalog()
                self.event(f"已删除 {len(selections)} 条轨迹")
                result = {}
            else:
                exported = self.store.export_selected(selections, command.destination)
                result = (
                    {"destination": str(exported)}
                    if command.destination is not None
                    else {"download": f"/api/exports/{exported.parent.name}"}
                )
                self.event(f"已导出 {len(selections)} 条轨迹")
            self.error, self.state = "", "IDLE"
            return result
        else:
            raise CaptureError("未知操作")

    def _publish(self):
        self.sequence += 1
        now = time.monotonic()
        ready, can_start, source_error = False, False, ""
        streams, positions = {}, {"state": [], "action": []}
        if self.sources:
            try:
                self.sources.check()
            except Exception as exc:
                source_error = str(exc)
            worker_errors = {
                worker.name: str(worker.error) for worker in self.sources.workers if worker.error is not None
            }
            owners = {camera.name: f"camera:{camera.name}" for camera in self.config.cameras}
            owners.update(
                {
                    f"{arm.name}.{field}.{group}": f"can:{port}"
                    for arm in self.config.arms
                    for field, port in (("state", arm.state_can), ("action", arm.command_can))
                    for group in ("12", "34", "56", "gripper")
                }
            )
            for name, stream in {**self.sources.vectors, **self.sources.cameras}.items():
                samples = stream.samples()
                age = now - samples[-1].time if samples else None
                hz = ((len(samples) - 1) / (samples[-1].time - samples[0].time)) if len(samples) > 1 else 0
                limit = (
                    self.config.max_camera_skew_s * 2
                    if name in self.sources.cameras
                    else self.config.max_vector_gap_s
                )
                held = ".action." in name and age is not None and age >= limit
                # Preview availability is local to its source; recording still checks every worker.
                stream_error = worker_errors.get(owners[name], worker_errors.get("simulation", ""))
                streams[name] = {
                    "age_s": age,
                    "hz": round(hz, 1) if age is not None and age < limit else 0,
                    "held": held,
                    "ready": not stream_error and age is not None and (age < limit or held),
                    "error": stream_error,
                }
            for field in positions:
                for arm in self.config.arms:
                    for group in ("12", "34", "56", "gripper"):
                        name = f"{arm.name}.{field}.{group}"
                        sample = self.sources.vectors[name].latest()
                        fresh = sample and streams[name]["ready"]
                        positions[field].extend(
                            sample.value.tolist() if fresh else [None] * (1 if group == "gripper" else 2)
                        )
            try:
                self.sources.check()
                self.sources.aligner.frame(now - self.config.alignment_delay_s, require_action=False)
                can_start = True
                self.sources.aligner.frame(now - self.config.alignment_delay_s)
                ready = True
            except Exception as exc:
                source_error = str(exc)
        writer = self.session.writer if self.session else None
        frames = writer.count if writer else 0
        state = self.state
        replay = self.replay.status() if self.replay else None
        if replay and replay["active"]:
            state = "REPLAYING"
        if state == "IDLE":
            state = (
                "REVIEW"
                if self.pending_rows
                else ("READY" if ready else "ARMABLE" if can_start else "WAITING")
            )
        snapshot = {
            "instance": self.instance,
            "sequence": self.sequence,
            "state": state,
            "ready": ready,
            "can_start": can_start and not self.replay_needs_reconnect,
            "replay": replay,
            "replay_needs_reconnect": self.replay_needs_reconnect,
            "awaiting_action": bool(writer and self.session.awaiting_action),
            "error": self.error,
            "source_error": source_error,
            "mock": self.mock,
            "task": self.session.task if self.session else self.config.task,
            "frames": frames,
            "duration_s": frames / self.config.fps,
            "queue_depth": writer.queue.qsize() if writer else 0,
            "fps": self.config.fps,
            "max_episode_s": self.config.max_episode_s,
            "min_episode_frames": self.config.min_episode_frames,
            "output": str(Path(self.config.output).expanduser().resolve()),
            "free_gb": round(shutil.disk_usage(self.store.root).free / 1e9, 1) if self.store else None,
            "cameras": [asdict(c) for c in self.config.cameras],
            "arms": [asdict(a) for a in self.config.arms],
            "streams": streams,
            "positions": positions,
            "names": self.config.names,
            "pending": self.pending_rows,
            "history": self.history_rows,
            "catalog_revision": self.catalog_revision(),
            "success_count": sum(r["outcome"] == "success" for r in self.history_rows),
            "failure_count": sum(r["outcome"] == "failure" for r in self.history_rows),
            "total_frames": sum(r["length"] for r in self.history_rows),
            "events": list(reversed(self.events)),
        }
        with self.lock:
            self.snapshot = snapshot

    def _preview(self):
        while not self.stop_event.is_set():
            sources = self.sources
            if sources:
                for name, stream in sources.cameras.items():
                    sample = stream.latest()
                    if sample is None:
                        continue
                    with self.lock:
                        cached = self.jpeg.get(name)
                    if cached and cached[0] == sample.time:
                        continue
                    try:
                        bgr = cv2.cvtColor(sample.value, cv2.COLOR_RGB2BGR)
                        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            with self.lock:
                                self.jpeg[name] = (sample.time, jpeg.tobytes())
                    except Exception:
                        LOGGER.exception("Preview encoding failed")
            self.stop_event.wait(0.1)

    def recording(self, kind: str, identifier: str) -> tuple[dict, Path, dict[str, Path]]:
        root = Path(self.config.output).expanduser().resolve()
        if kind in {"pending", "failed"}:
            if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
                raise ValueError("无效轨迹 ID")
            directory = root / ".piper" / kind / identifier
            metadata = read_json(directory / "manifest.json")
            data = directory / "frames.parquet"
            videos = {c.name: directory / f"{c.key}.mp4" for c in self.config.cameras}
        elif kind == "saved" and identifier.isdigit():
            index = int(identifier)
            metadata = read_json(root / ".piper/committed" / f"{index:06d}.json")
            variables = {"episode_index": index, "episode_chunk": index // 1000}
            data = root / DATA_PATH.format(**variables)
            videos = {
                c.name: root / VIDEO_PATH.format(video_key=c.key, **variables) for c in self.config.cameras
            }
        else:
            raise ValueError("无效轨迹 ID")
        return metadata, data, videos


def create_app(config: CaptureConfig, *, mock: bool = False, resume: bool = False) -> FastAPI:
    runtime = WebRuntime(config, mock=mock, resume=resume)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.close)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.runtime = runtime

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        # Local-only UI: reject foreign Host headers and protect mutations from cross-origin forms.
        hostname = request.url.hostname
        if hostname not in {"127.0.0.1", "localhost", "::1", "testserver"}:
            return JSONResponse({"detail": "仅允许本机访问"}, status_code=403)
        if request.method == "POST" and request.headers.get("x-capture-token") != token:
            return JSONResponse({"detail": "操作令牌失效，请刷新页面"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; media-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/api/status")
    def status():
        return JSONResponse({**runtime.status(), "token": token}, headers={"Cache-Control": "no-store"})

    @app.post("/api/command")
    async def command(body: Command):
        try:
            result = await asyncio.wrap_future(runtime.submit(body))
            if "status" in result:
                result["status"]["token"] = token
            return result
        except CaptureError as exc:
            return JSONResponse(
                {"detail": str(exc), "status": {**runtime.status(), "token": token}}, status_code=409
            )

    @app.get("/api/cameras/{name}.jpg")
    def camera(name: str):
        if name not in {c.name for c in config.cameras}:
            raise HTTPException(404, "未知相机")
        with runtime.lock:
            jpeg = runtime.jpeg.get(name)
        if not jpeg or time.monotonic() - jpeg[0] > 1:
            raise HTTPException(503, "相机画面暂不可用")
        return Response(jpeg[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/recordings/{kind}/{identifier}")
    def recording(kind: str, identifier: str):
        try:
            with runtime.catalog_lock:
                metadata, data, videos = runtime.recording(kind, identifier)
                table = pq.read_table(data, columns=["observation.state", "action"]).to_pydict()
            n = metadata["length"]
            indices = np.unique(np.linspace(0, n - 1, min(n, 600), dtype=int))
            return {
                "id": identifier,
                "kind": kind,
                "task": metadata["task"],
                "length": n,
                "fps": config.fps,
                "issues": metadata["issues"],
                "outcome": metadata.get("outcome"),
                "recording_revision": recording_revision(metadata),
                "names": config.names,
                "videos": {name: f"/api/recordings/{kind}/{identifier}/video/{name}" for name in videos},
                "chart": {
                    "indices": indices.tolist(),
                    "state": np.asarray(table["observation.state"])[indices].tolist(),
                    "action": np.asarray(table["action"])[indices].tolist(),
                },
            }
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(404, "轨迹不存在或录制未完成") from exc

    @app.get("/api/recordings/{kind}/{identifier}/video/{name}")
    def video(kind: str, identifier: str, name: str):
        try:
            _, _, videos = runtime.recording(kind, identifier)
            path = videos[name]
            if not path.is_file():
                raise FileNotFoundError(path)
            return FileResponse(path, media_type="video/mp4")
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(404, "视频不存在") from exc

    @app.get("/api/exports/{identifier}")
    def export(identifier: str):
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise HTTPException(404, "导出文件不存在")
        directory = Path(config.output).expanduser().resolve() / ".piper/exports" / identifier
        archive = directory / "recordings.zip"
        if not archive.is_file():
            raise HTTPException(404, "导出文件不存在")
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=f"piper-recordings-{identifier[:8]}.zip",
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def recording_revision(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def serve(args):
    from dataclasses import replace

    import uvicorn

    from .config import load_config

    config = load_config(args.config)
    if args.output:
        config = replace(config, output=str(Path(args.output).expanduser().resolve()))
    elif args.mock:
        config = replace(config, output=str(Path("work_dirs/web_simulation").resolve()))
    config.validate()
    # Only resume this collector's own datasets, with the existing contract validation and writer lock.
    resume = (Path(config.output) / ".piper/contract.json").is_file()
    uvicorn.run(
        create_app(config, mock=args.mock, resume=resume),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="info",
    )
