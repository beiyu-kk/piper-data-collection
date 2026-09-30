from __future__ import annotations

import math
import threading
import time
from contextlib import ExitStack

import cv2
import numpy as np

from .config import CameraConfig, CaptureConfig
from .timeline import Aligner, CaptureError, Sample, Timeline

JOINT_SCALE = math.pi / 180_000.0
GRIPPER_SCALE = 1e-6
CAN_GROUPS = {
    0x2A5: ("state", "12", (1, 2)),
    0x2A6: ("state", "34", (3, 4)),
    0x2A7: ("state", "56", (5, 6)),
    0x2A8: ("state", "gripper", ()),
    0x155: ("action", "12", (1, 2)),
    0x156: ("action", "34", (3, 4)),
    0x157: ("action", "56", (5, 6)),
    0x159: ("action", "gripper", ()),
}


class SdkDecoder:
    """Use the vendor protocol parser without initializing an arm or sending CAN."""

    def __init__(self):
        from piper_sdk.piper_msgs.msg_v2 import PiperMessage
        from piper_sdk.protocol.protocol_v2 import C_PiperParserV2

        self.parser = C_PiperParserV2()
        self.message_type = PiperMessage

    def decode(self, message) -> tuple[str, str, np.ndarray] | None:
        if message.is_error_frame:
            raise CaptureError("CAN error frame received")
        if message.is_extended_id or message.is_remote_frame or message.arbitration_id not in CAN_GROUPS:
            return None
        if len(message.data) != 8:
            raise CaptureError(f"invalid CAN payload length for 0x{message.arbitration_id:x}")
        decoded = self.message_type()
        if not self.parser.DecodeMessage(message, decoded):
            raise CaptureError("SDK failed to decode a required CAN frame")
        field, group, joints = CAN_GROUPS[message.arbitration_id]
        if group == "gripper":
            data = decoded.gripper_feedback if field == "state" else decoded.arm_gripper_ctrl
            values = [data.grippers_angle * GRIPPER_SCALE]
        else:
            data = decoded.arm_joint_feedback if field == "state" else decoded.arm_joint_ctrl
            values = [getattr(data, f"joint_{joint}") * JOINT_SCALE for joint in joints]
        return field, group, np.asarray(values, dtype=np.float32)


class Worker:
    def __init__(self, name: str):
        self.name = name
        self.stop = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)

    def _run(self):
        try:
            self.run()
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc

    def run(self):
        raise NotImplementedError

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise CaptureError(f"{self.name}: source worker did not stop")


class CanSource(Worker):
    def __init__(self, channel: str, routes: dict[tuple[str, str], Timeline]):
        super().__init__(f"can:{channel}")
        self.channel = channel
        self.routes = routes
        self.decoder = SdkDecoder()
        self.bus = None

    def run(self):
        import can

        ids = [
            identifier
            for identifier, (field, group, _) in CAN_GROUPS.items()
            if (field, group) in self.routes
        ]
        # Python never sends on this socket. Hardware master/follower operation is external.
        with can.Bus(
            interface="socketcan",
            channel=self.channel,
            receive_own_messages=False,
            can_filters=[{"can_id": i, "can_mask": 0x7FF, "extended": False} for i in ids],
        ) as bus:
            self.bus = bus
            last_receive = time.monotonic()
            while not self.stop.is_set():
                message = bus.recv(timeout=0.1)
                if message is None:
                    if time.monotonic() - last_receive > 0.5:
                        raise CaptureError("CAN bus stopped receiving data")
                    continue
                last_receive = time.monotonic()
                result = self.decoder.decode(message)
                if result is None:
                    continue
                field, group, value = result
                # Recompute the wall/monotonic offset to tolerate gradual wall-clock correction.
                age = time.time() - message.timestamp
                timestamp = last_receive - age
                if abs(age) > 0.5:
                    raise CaptureError("CAN receive timestamp drift or backlog exceeds 0.5 seconds")
                stream = self.routes[field, group]
                previous = stream.latest()
                if previous and timestamp <= previous.time:
                    timestamp = previous.time + 1e-6
                stream.append(Sample(timestamp, value))


class CameraSource(Worker):
    def __init__(self, config: CameraConfig, timeline: Timeline):
        super().__init__(f"camera:{config.name}")
        self.config = config
        self.timeline = timeline

    def run(self):
        if self.config.backend == "realsense":
            self._realsense()
        else:
            self._opencv()

    def _realsense(self):
        import pyrealsense2 as rs

        c = self.config
        pipeline = rs.pipeline()
        options = rs.config()
        options.enable_device(c.serial)
        options.enable_stream(rs.stream.color, c.width, c.height, rs.format.rgb8, c.fps)
        profile = pipeline.start(options)
        try:
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
            while not self.stop.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                timestamp = time.monotonic()
                color = frames.get_color_frame()
                if not color:
                    raise CaptureError(f"{c.name}: missing color frame")
                rgb = np.asanyarray(color.get_data()).copy()
                self.timeline.append(Sample(timestamp, rgb, color.get_frame_number(), color.get_timestamp()))
        finally:
            pipeline.stop()

    def _opencv(self):
        c = self.config
        capture = cv2.VideoCapture(c.device, cv2.CAP_V4L2)
        try:
            if not capture.isOpened():
                raise CaptureError(f"cannot open V4L2 camera {c.device}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
            capture.set(cv2.CAP_PROP_FPS, c.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            sequence = 0
            while not self.stop.is_set():
                ok, bgr = capture.read()
                timestamp = time.monotonic()
                if not ok:
                    raise CaptureError(f"{c.name}: camera read failed")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.timeline.append(Sample(timestamp, rgb, sequence))
                sequence += 1
        finally:
            capture.release()


class MockSource(Worker):
    def __init__(self, config: CaptureConfig, vectors: dict, cameras: dict):
        super().__init__("simulation")
        self.config, self.vectors, self.cameras = config, vectors, cameras

    def run(self):
        start = time.monotonic()
        next_images = {c.name: start for c in self.config.cameras}
        sequences = {c.name: 0 for c in self.config.cameras}
        while not self.stop.is_set():
            now = time.monotonic()
            t = now - start
            for arm in self.config.arms:
                for field in ("state", "action"):
                    phase = t - (0.08 if field == "state" else 0)
                    q = np.asarray(
                        [
                            base + 0.05 * math.sin(phase + i * 0.3)
                            for i, base in enumerate((0, 0.5, -0.5, 0, 0, 0))
                        ]
                        + [0.04 + 0.015 * math.sin(phase)],
                        dtype=np.float32,
                    )
                    for group, values in (("12", q[:2]), ("34", q[2:4]), ("56", q[4:6]), ("gripper", q[6:])):
                        self.vectors[f"{arm.name}.{field}.{group}"].append(Sample(now, values.copy()))
            for index, c in enumerate(self.config.cameras):
                if now < next_images[c.name]:
                    continue
                sequence = sequences[c.name]
                rgb = np.full((c.height, c.width, 3), (36, 44, 50), dtype=np.uint8)
                rgb[:, : c.width // 3] = (180, 65, 55)
                rgb[:, c.width // 3 : 2 * c.width // 3] = (65, 155, 100)
                rgb[:, 2 * c.width // 3 :] = (65, 100, 190)
                x = int((math.sin(t + index) + 1) / 2 * (c.width - 60))
                cv2.rectangle(rgb, (x, c.height // 2), (x + 55, c.height // 2 + 40), (245, 240, 210), -1)
                cv2.putText(
                    rgb, f"SIMULATION {c.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                cv2.putText(
                    rgb, f"{sequence:06d}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                self.cameras[c.name].append(Sample(now, rgb, sequence))
                sequences[c.name] += 1
                next_images[c.name] = now + 1 / c.fps
            self.stop.wait(0.005)


class Sources:
    def __init__(self, config: CaptureConfig, *, mock: bool = False):
        self.config = config
        self.mock = mock
        self.vectors = {
            f"{a.name}.{field}.{group}": Timeline(int(config.history_s * 1000))
            for a in config.arms
            for field in ("state", "action")
            for group in ("12", "34", "56", "gripper")
        }
        self.cameras = {c.name: Timeline(max(4, int(config.history_s * c.fps))) for c in config.cameras}
        self.aligner = Aligner(config, self.vectors, self.cameras)
        self.workers: list[Worker] = []
        self._stack = ExitStack()

    def __enter__(self):
        if self.mock:
            self.workers = [MockSource(self.config, self.vectors, self.cameras)]
        else:
            routes: dict = {}
            for arm in self.config.arms:
                for field, port in (("state", arm.state_can), ("action", arm.command_can)):
                    for group in ("12", "34", "56", "gripper"):
                        routes.setdefault(port, {})[field, group] = self.vectors[
                            f"{arm.name}.{field}.{group}"
                        ]
            self.workers = [CanSource(port, route) for port, route in routes.items()]
            self.workers.extend(CameraSource(c, self.cameras[c.name]) for c in self.config.cameras)
        try:
            for worker in self.workers:
                worker.thread.start()
                self._stack.callback(worker.close)
        except BaseException:
            self._stack.close()
            raise
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def check(self):
        for worker in self.workers:
            if worker.error is not None:
                raise CaptureError(f"{worker.name}: {worker.error}") from worker.error

    def wait_ready(self, timeout: float = 8.0, *, require_action: bool = True):
        deadline = time.monotonic() + timeout
        last_error = "waiting for streams"
        while time.monotonic() < deadline:
            self.check()
            try:
                return self.aligner.frame(
                    time.monotonic() - self.config.alignment_delay_s, require_action=require_action
                )
            except CaptureError as exc:
                last_error = str(exc)
            time.sleep(0.03)
        raise CaptureError(f"sources not ready after {timeout}s: {last_error}")

    def status(self) -> dict:
        now = time.monotonic()
        return {
            name: {
                "samples": len(stream.samples()),
                "age_s": None if stream.latest() is None else round(now - stream.latest().time, 4),
            }
            for name, stream in {**self.vectors, **self.cameras}.items()
        }
