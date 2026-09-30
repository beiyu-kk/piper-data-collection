from __future__ import annotations

import argparse
import json
import math
import select
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .config import load_config
from .session import CaptureSession
from .sources import Sources
from .storage import DatasetStore, read_json
from .timeline import CaptureError
from .validate import validate_dataset


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def preview(sources: Sources, label: str) -> str:
    images = []
    for name, stream in sources.cameras.items():
        sample = stream.latest()
        if sample is None:
            continue
        bgr = cv2.cvtColor(sample.value, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (480, 360))
        cv2.putText(bgr, name, (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        images.append(bgr)
    if images:
        canvas = np.concatenate(images, axis=1)
        cv2.putText(canvas, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Piper Capture", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            return chr(key)
        if cv2.getWindowProperty("Piper Capture", cv2.WND_PROP_VISIBLE) < 1:
            return "q"
    return ""


def record(args):
    config = load_config(args.config)
    if args.output:
        config = replace(config, output=str(Path(args.output).expanduser().resolve()))
    elif args.mock:
        config = replace(config, output=str(Path("work_dirs/simulation").resolve()))
    if args.task:
        config = replace(config, task=args.task)
    config.validate()
    if args.episodes is not None:
        if args.episodes <= 0 or not math.isfinite(args.duration) or args.duration <= 0:
            raise ValueError("episodes and duration must be positive")
        frames = int(args.duration * config.fps)
        if not config.min_episode_frames <= frames <= config.max_episode_s * config.fps:
            raise ValueError("duration lies outside configured episode length limits")
    elif not sys.stdin.isatty() and not args.preview:
        raise ValueError(
            "interactive recording needs a terminal; use --episodes and --duration for batch mode"
        )
    with (
        DatasetStore(config, resume=args.resume, mock=args.mock) as store,
        Sources(config, mock=args.mock) as sources,
    ):
        sources.wait_ready()
        session = CaptureSession(sources, store)
        print_json(
            {
                "state": "READY",
                "output": str(store.root),
                "mock": args.mock,
                "existing_episodes": len(store.records()),
                "pending": [r["id"] for r in store.pending()],
            }
        )
        try:
            if args.episodes is not None:
                for _ in range(args.episodes):
                    session.start()
                    while session.writer.count < frames:
                        session.tick()
                        if args.preview and preview(sources, f"RECORDING {session.writer.count}") == "q":
                            raise KeyboardInterrupt
                        time.sleep(0.002)
                    directory = session.stop()
                    index = store.accept(directory)
                    print_json({"state": "SAVED", "episode_index": index, "frames": frames})
            else:
                interactive(session, args.preview)
        except BaseException as exc:
            if session.writer is not None:
                directory = session.stop(failure=f"capture interrupted: {type(exc).__name__}: {exc}")
                print_json({"state": "FAILED", "pending": directory.name, "error": str(exc)})
            raise
        finally:
            if args.preview:
                cv2.destroyAllWindows()


def interactive(session: CaptureSession, show_preview: bool):
    print(
        "Commands: r=start, s=stop, v=replay, a=accept, d=discard, t <task>=set task, q=quit. Enter in terminal."
    )
    pending = None
    next_preview = 0.0
    while True:
        command = ""
        try:
            if session.writer is not None and session.tick():
                command = "s"
        except CaptureError as exc:
            pending = session.stop(failure=str(exc))
            print_json({"state": "FAILED", "pending": pending.name, "error": str(exc)})
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            command = line.strip() if line else "q"
        if show_preview and time.monotonic() >= next_preview:
            label = (
                f"RECORDING {session.writer.count}" if session.writer else ("REVIEW" if pending else "READY")
            )
            command = preview(session.sources, label) or command
            next_preview = time.monotonic() + 1 / 15
        if command:
            try:
                if command == "q":
                    if session.writer:
                        pending = session.stop()
                    if pending:
                        print_json({"state": "PENDING", "id": pending.name})
                    return
                if command == "r":
                    if pending:
                        raise CaptureError("accept or discard the pending episode before recording")
                    session.start()
                    print_json({"state": "RECORDING", "task": session.task})
                elif command == "s":
                    pending = session.stop()
                    manifest = read_json(pending / "manifest.json")
                    print_json(
                        {
                            "state": "REVIEW",
                            "id": pending.name,
                            "frames": manifest["length"],
                            "issues": manifest["issues"],
                        }
                    )
                elif command == "a":
                    if not pending:
                        raise CaptureError("no episode is pending review")
                    index = session.store.accept(pending)
                    pending = None
                    print_json({"state": "SAVED", "episode_index": index})
                elif command == "v":
                    if not pending or session.writer:
                        raise CaptureError("stop recording before replaying a pending episode")
                    replay(SimpleNamespace(root=str(session.store.root), pending=pending.name, episode=0))
                elif command == "d":
                    if session.writer:
                        pending = session.stop()
                    if not pending:
                        raise CaptureError("no episode to discard")
                    path = session.store.reject(pending)
                    pending = None
                    print_json({"state": "DISCARDED", "retained_at": str(path)})
                elif command.startswith("t "):
                    if session.writer or pending:
                        raise CaptureError("task can only change in READY state")
                    session.task = command[2:].strip()
                    if not session.task:
                        raise ValueError("task must not be empty")
                    print_json({"task": session.task})
            except (CaptureError, ValueError) as exc:
                print_json({"error": str(exc)})
        time.sleep(0.002)


def doctor(args):
    config = load_config(args.config)
    with Sources(config, mock=args.mock) as sources:
        try:
            frame = sources.wait_ready()
        except CaptureError:
            print_json({"state": "NOT_READY", "streams": sources.status()})
            raise
        print_json(
            {
                "state": "READY",
                "mock": args.mock,
                "state_vector": frame.state.tolist(),
                "leader_command": frame.action.tolist(),
                "streams": sources.status(),
                "camera_timing": frame.timing["cameras"],
            }
        )


def live_preview(args):
    config = load_config(args.config)
    with Sources(config, mock=args.mock) as sources:
        try:
            while True:
                sources.check()
                try:
                    sources.aligner.frame(time.monotonic() - config.alignment_delay_s)
                    label = "READY"
                except CaptureError:
                    label = "WAITING FOR STREAMS"
                if preview(sources, label) == "q":
                    return
                time.sleep(1 / 30)
        finally:
            cv2.destroyAllWindows()


def devices():
    result = subprocess.run(
        ["ip", "-details", "-json", "link", "show", "type", "can"], capture_output=True, text=True, check=True
    )
    print_json({"can": json.loads(result.stdout)})
    try:
        import pyrealsense2 as rs
    except ImportError:
        print_json({"realsense": "pyrealsense2 is not installed"})
        return
    print_json(
        {
            "realsense": [
                {"name": d.get_info(rs.camera_info.name), "serial": d.get_info(rs.camera_info.serial_number)}
                for d in rs.context().query_devices()
            ]
        }
    )


def pending_command(args):
    config = load_config(args.config)
    if args.output:
        config = replace(config, output=args.output)
    elif args.mock:
        config = replace(config, output=str(Path("work_dirs/simulation").resolve()))
    with DatasetStore(config, resume=True, mock=args.mock) as store:
        if args.accept:
            print_json({"saved_episode": store.accept(store.pending_path(args.accept))})
        elif args.discard:
            print_json({"discarded": str(store.reject(store.pending_path(args.discard)))})
        else:
            print_json([{k: r.get(k) for k in ("id", "task", "length", "issues")} for r in store.pending()])


def replay(args):
    root = Path(args.root).expanduser().resolve()
    if args.pending:
        if len(args.pending) != 32 or any(c not in "0123456789abcdef" for c in args.pending):
            raise ValueError("invalid pending recording ID")
        directory = root / ".piper/pending" / args.pending
        manifest = read_json(directory / "manifest.json")
        fps = manifest["contract"]["fps"]
        paths = {key: directory / f"{key}.mp4" for key in manifest["contract"]["cameras"]}
    else:
        info = read_json(root / "meta/info.json")
        if not 0 <= args.episode < info["total_episodes"]:
            raise ValueError("episode index is out of range")
        fps = info["fps"]
        paths = {
            key: root
            / info["video_path"].format(
                video_key=key, episode_index=args.episode, episode_chunk=args.episode // info["chunks_size"]
            )
            for key, feature in info["features"].items()
            if feature["dtype"] == "video"
        }
    captures = {key: cv2.VideoCapture(str(path)) for key, path in paths.items()}
    try:
        if not all(c.isOpened() for c in captures.values()):
            raise CaptureError("cannot open all replay videos")
        index, paused = 0, False
        while True:
            if not paused:
                images = []
                for key, capture in captures.items():
                    ok, bgr = capture.read()
                    if not ok:
                        return
                    bgr = cv2.resize(bgr, (480, 360))
                    cv2.putText(
                        bgr,
                        f"{key.rsplit('.', 1)[-1]} {index}",
                        (10, 340),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    images.append(bgr)
                cv2.imshow("Piper Replay", np.concatenate(images, axis=1))
                index += 1
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == ord("q") or cv2.getWindowProperty("Piper Replay", cv2.WND_PROP_VISIBLE) < 1:
                return
            if key == ord(" "):
                paused = not paused
    finally:
        for capture in captures.values():
            capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Piper leader/follower capture to LeRobot v2.1")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="list CAN interfaces and RealSense serials")
    p = commands.add_parser("web", help="local browser collection console")
    p.add_argument("--config", default="configs/right_arm.yaml")
    p.add_argument("--output")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    for name in ("doctor", "preview", "record", "pending"):
        p = commands.add_parser(name)
        p.add_argument("--config", default="configs/right_arm.yaml")
        p.add_argument("--mock", action="store_true")
        if name in ("record", "pending"):
            p.add_argument("--output")
        if name == "record":
            p.add_argument("--task")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--preview", action="store_true")
            p.add_argument("--episodes", type=int)
            p.add_argument("--duration", type=float, default=10.0)
        if name == "pending":
            group = p.add_mutually_exclusive_group()
            group.add_argument("--accept", metavar="ID")
            group.add_argument("--discard", metavar="ID")
    p = commands.add_parser("validate")
    p.add_argument("root", type=Path)
    p.add_argument("--metadata-only", action="store_true")
    p = commands.add_parser("replay")
    p.add_argument("root")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--pending")
    args = parser.parse_args()
    try:
        if args.command == "devices":
            devices()
        elif args.command == "web":
            from .web import serve

            serve(args)
        elif args.command == "doctor":
            doctor(args)
        elif args.command == "preview":
            live_preview(args)
        elif args.command == "record":
            record(args)
        elif args.command == "pending":
            pending_command(args)
        elif args.command == "replay":
            replay(args)
        elif args.command == "validate":
            print_json(validate_dataset(args.root, decode_videos=not args.metadata_only))
    except KeyboardInterrupt:
        print("Stopped. Incomplete recordings are retained under .piper/pending.", file=sys.stderr)
        raise SystemExit(130)
    except (CaptureError, ValueError, FileNotFoundError, FileExistsError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
