from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from .sources import GRIPPER_SCALE, JOINT_SCALE, SdkDecoder
from .timeline import CaptureError

START_VELOCITY = np.array([0.175] * 6 + [0.015])
START_TOLERANCE = np.array([0.10] * 6 + [0.005])
TRACK_TOLERANCE = np.array([0.25] * 6 + [0.015])
TRACK_LAG_S = 0.15
TRACK_PERSIST_S = 0.2
TRACK_HARD_FACTOR = 2.0
START_SETTLE_S = 0.2
START_TIMEOUT_S = 3.0
END_HOLD_S = 0.5
LOGGER = logging.getLogger(__name__)
CONTROL_IDS = {
    base + offset for base in (0x150, 0x151, 0x155, 0x156, 0x157, 0x159) for offset in (0, 0x10, 0x20)
} | {0x152, 0x153, 0x154, 0x470, 0x471}


def encode_positions(points, with_gripper, *, context="轨迹"):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] != 7 or not np.isfinite(values).all():
        raise CaptureError("回放位置维度或数值无效")
    dimensions = 7 if with_gripper else 6
    scaled = values[..., :dimensions].copy()
    # Preserve recorded positions, checking only representability in the protocol's signed int32 fields.
    with np.errstate(over="ignore"):
        scaled[..., :6] /= JOINT_SCALE
        if with_gripper:
            scaled[..., 6] /= GRIPPER_SCALE
        wire = np.rint(scaled)
    invalid = (wire < -(2**31)) | (wire > 2**31 - 1)
    if invalid.any():
        rows, mask = np.atleast_2d(values), np.atleast_2d(invalid)
        details = []
        for axis in np.flatnonzero(mask.any(axis=0)):
            first = int(np.flatnonzero(mask[:, axis])[0])
            factor, unit = (1000, "mm") if axis == 6 else (1, "rad")
            name = "夹爪" if axis == 6 else f"J{axis + 1}"
            details.append(f"{name} 第 {first} 帧 {rows[first, axis] * factor:.6g} {unit}")
        raise CaptureError(
            f"{context}超出 CAN int32 编码范围：" + "；".join(details) + "。未裁剪原值，未发送该目标"
        )
    return wire.astype(np.int64)


@dataclass
class ReplayPlan:
    points: np.ndarray
    times: np.ndarray
    field: str
    speed: float
    token: str
    created: float
    with_gripper: bool = False

    @classmethod
    def load(cls, path, metadata, config, field, speed, with_gripper=False):
        if len(config.arms) != 1 or config.arms[0].name != "right":
            raise CaptureError("真机回放目前仅支持单个右从臂")
        if metadata.get("issues") or metadata.get("contract") != config.contract():
            raise CaptureError("轨迹质量检查未通过或单位/配置不匹配，不能真机回放")
        if field not in {"observation.state", "action"} or speed not in {0.25, 0.5, 1.0}:
            raise CaptureError("无效回放来源或速度")
        table = pq.read_table(path, columns=[field, "timestamp"]).to_pydict()
        points = np.asarray(table[field], dtype=np.float64)
        timestamps = np.asarray(table["timestamp"], dtype=np.float64)
        if (
            points.shape != (metadata["length"], 7)
            or len(points) < 2
            or not np.isfinite(points).all()
            or not np.isfinite(timestamps).all()
            or not np.allclose(timestamps, np.arange(len(points)) / config.fps, atol=1e-4)
        ):
            raise CaptureError("轨迹维度、数值或时间戳无效")
        encode_positions(points, with_gripper)
        # Keep the recorded timeline; only the operator-selected playback rate scales time.
        times = (timestamps - timestamps[0]) / speed
        return cls(points, times, field, speed, secrets.token_hex(16), time.monotonic(), with_gripper)

    def describe(self):
        return {
            "plan_token": self.token,
            "field": self.field,
            "speed": self.speed,
            "with_gripper": self.with_gripper,
            "position_policy": "recorded_without_clipping",
            "timing_policy": "recorded_timestamps",
            "tracking_policy": "recent_sent_targets",
            "tracking_lag_s": TRACK_LAG_S,
            "tracking_persist_s": TRACK_PERSIST_S,
            "frames": len(self.points),
            "trajectory_duration_s": float(self.times[-1]),
            "end_hold_s": END_HOLD_S,
            "duration_s": float(self.times[-1] + END_HOLD_S),
            "start_mode": "move_to_first_pose",
            "max_start_joint_speed": float(START_VELOCITY[0]),
            "max_start_gripper_speed": float(START_VELOCITY[-1]),
            "first_pose": self.points[0].tolist(),
            "last_pose": self.points[-1].tolist(),
        }

    def at(self, elapsed):
        return np.array([np.interp(elapsed, self.times, self.points[:, i]) for i in range(7)])

    def start_duration(self, pose):
        encode_positions(pose, self.with_gripper, context="回起点起始位置")
        dimensions = 7 if self.with_gripper else 6
        difference = np.abs(self.points[0, :dimensions] - np.asarray(pose)[:dimensions])
        # Cubic smoothstep has a peak slope of 1.5 and zero velocity at both ends.
        return max(0.5, float(1.5 * (difference / START_VELOCITY[:dimensions]).max()))


class SocketReplayTransport:
    """SDK protocol codec on one checked SocketCAN socket; no SDK initialization side effects."""

    def __init__(self, channel, first_pose, with_gripper=False):
        import can
        from piper_sdk import piper_msgs

        self.msgs = piper_msgs.msg_v2
        self.decoder = SdkDecoder()
        self.bus = can.Bus(interface="socketcan", channel=channel, receive_own_messages=False)
        self.states, self.health = {}, {}
        self.health_details = {}
        self.closed = False
        self.with_gripper = with_gripper

    def poll(self):
        for _ in range(2000):
            message = self.bus.recv(timeout=0)
            if message is None:
                return
            if message.is_error_frame:
                raise CaptureError("回放 CAN 总线错误")
            if message.is_extended_id or message.is_remote_frame:
                continue
            if message.arbitration_id in CONTROL_IDS:
                raise CaptureError("检测到其他控制指令，请断开主臂并关闭其他控制程序")
            if abs(time.time() - message.timestamp) > 0.2:
                raise CaptureError("回放 CAN 接收积压")
            decoded = self.decoder.decode(message)
            now = time.monotonic()
            if decoded:
                field, group, values = decoded
                if field == "state":
                    self.states[group] = (now, values)
                    if group == "gripper":
                        self.health[0x2A8] = (now, message.data[6] & 0x7F == 0x40)
                        self.health_details[0x2A8] = f"夹爪未使能或故障（状态码 0x{message.data[6]:02X}）"
            identifier = message.arbitration_id
            if identifier == 0x2A1 or 0x261 <= identifier <= 0x266:
                msg = self.decoder.message_type()
                if len(message.data) != 8 or not self.decoder.parser.DecodeMessage(message, msg):
                    raise CaptureError("回放反馈解析失败")
                if identifier == 0x2A1:
                    status = msg.arm_status_msgs
                    healthy = status.arm_status == 0 and status.err_code == 0
                    detail = f"arm_status=0x{int(status.arm_status):02X}，err_code=0x{status.err_code:04X}"
                    if status.arm_status == 1:
                        self.health_details[identifier] = (
                            f"从臂处于急停状态（{detail}）。请先在设备端解除急停，再重新预检回放；"
                            "网页重新连接不会解除急停"
                        )
                    else:
                        self.health_details[identifier] = f"从臂控制器状态异常（{detail}）"
                else:
                    # Exactly the enable bit, with every fault bit clear.
                    code = getattr(msg, f"arm_low_spd_feedback_{identifier - 0x260}").foc_status_code
                    healthy = code == 64
                    self.health_details[identifier] = (
                        f"J{identifier - 0x260} 未使能或驱动故障（状态码 0x{code:02X}）"
                    )
                self.health[identifier] = (now, healthy)
        raise CaptureError("回放 CAN 接收积压")

    def feedback(self):
        self.poll()
        now = time.monotonic()
        if any(g not in self.states or now - self.states[g][0] > 0.15 for g in ("12", "34", "56", "gripper")):
            raise CaptureError("缺少新鲜的从臂关节或夹爪反馈")
        required_health = (0x2A1, *range(0x261, 0x267), *((0x2A8,) if self.with_gripper else ()))
        missing = [i for i in required_health if i not in self.health or now - self.health[i][0] > 0.5]
        if missing:
            raise CaptureError("从臂状态反馈缺失或过期：" + "、".join(f"0x{i:03X}" for i in missing))
        unhealthy = [i for i in required_health if not self.health[i][1]]
        if unhealthy:
            raise CaptureError("；".join(self.health_details[i] for i in unhealthy))
        return np.concatenate([self.states[g][1] for g in ("12", "34", "56", "gripper")])

    def _send(self, kind, **kwargs):
        import can

        message = self.msgs.PiperMessage(type_=getattr(self.msgs.ArmMsgType, kind), **kwargs)
        packet = can.Message(is_extended_id=False)
        if not self.decoder.parser.EncodeMessage(message, packet):
            raise CaptureError("回放控制帧编码失败")
        self.bus.send(packet, timeout=0.02)

    def begin(self, pose):
        # Preserve the hardware following mode: no reset, homing, enable, or master/slave writes.
        self.target(pose)
        # The SDK documents that the speed percentage is ignored in high-follow mode (0xAD).
        self._send("PiperMsgMotionCtrl_2", arm_motion_ctrl_2=self.msgs.ArmMsgMotionCtrl_2(1, 1, 10, 0xAD))

    def target(self, pose):
        values = encode_positions(pose, self.with_gripper, context="发送目标").tolist()
        joints = self.msgs.ArmMsgJointCtrl(*values[:6])
        for kind in ("PiperMsgJointCtrl_12", "PiperMsgJointCtrl_34", "PiperMsgJointCtrl_56"):
            self._send(kind, arm_joint_ctrl=joints)
        if self.with_gripper:
            self._send(
                "PiperMsgGripperCtrl",
                arm_gripper_ctrl=self.msgs.ArmMsgGripperCtrl(values[6], 500, 1, 0),
            )

    def halt(self):
        # Attempt the joint stop even when a prior transmit failed; never reset or release joints.
        try:
            self._send("PiperMsgMotionCtrl_1", arm_motion_ctrl_1=self.msgs.ArmMsgMotionCtrl_1(1, 0, 0))
        finally:
            grip = self.states.get("gripper")
            if self.with_gripper and grip and time.monotonic() - grip[0] < 0.15:
                opening = int(encode_positions([0] * 6 + [grip[1][0]], True, context="夹爪保持位置")[6])
                self._send(
                    "PiperMsgGripperCtrl",
                    arm_gripper_ctrl=self.msgs.ArmMsgGripperCtrl(opening, 500, 1, 0),
                )

    def close(self):
        if not self.closed:
            self.closed = True
            self.bus.shutdown()


class MockReplayTransport:
    def __init__(self, channel, first_pose, with_gripper=False):
        self.pose = first_pose.copy()

    def poll(self):
        pass

    def feedback(self):
        return self.pose.copy()

    def begin(self, pose):
        self.target(pose)

    def target(self, pose):
        self.pose = pose.copy()

    def halt(self):
        pass

    def close(self):
        pass


class TrackingGuard:
    """Compare feedback to recent, successfully sent whole-arm targets."""

    def __init__(self, with_gripper):
        self.dimensions = 7 if with_gripper else 6
        self.history = deque()
        self.exceeded_since = None

    def sent(self, pose, now):
        self.history.append((now, np.asarray(pose).copy()))

    def check(self, measured, now):
        if not self.history or now - self.history[-1][0] > TRACK_LAG_S:
            raise CaptureError("回放目标发送中断，已停止")
        while len(self.history) > 1 and self.history[0][0] < now - TRACK_LAG_S:
            self.history.popleft()
        targets = np.array([pose for _, pose in self.history])
        errors = np.abs(targets - measured)
        ratios = (errors[:, : self.dimensions] / TRACK_TOLERANCE[: self.dimensions]).max(axis=1)
        # Match one complete pose; independently matching each axis could hide a wrong configuration.
        index = int(np.argmin(ratios))
        outside = ratios[index] > 1.0
        if not outside:
            self.exceeded_since = None
        elif self.exceeded_since is None:
            self.exceeded_since = now
        duration = 0.0 if self.exceeded_since is None else now - self.exceeded_since
        reason = None
        if ratios[index] > TRACK_HARD_FACTOR:
            reason = "large_deviation"
        elif outside and duration >= TRACK_PERSIST_S:
            reason = "persistent_deviation"
        return {
            "reference_target": targets[index].tolist(),
            "last_sent_target": targets[-1].tolist(),
            "feedback": measured.tolist(),
            "absolute_error": errors[index].tolist(),
            "tolerance": TRACK_TOLERANCE.tolist(),
            "immediate_tolerance": (TRACK_TOLERANCE * TRACK_HARD_FACTOR).tolist(),
            "reference_age_s": float(now - self.history[index][0]),
            "outside_for_s": float(duration),
            "exceeded_axes": np.flatnonzero(
                errors[index, : self.dimensions] > TRACK_TOLERANCE[: self.dimensions]
            ).tolist(),
            "stop_reason": reason,
        }


def tracking_error(detail):
    stage = "回起点" if detail["stage"] == "returning" else "回放"
    reason = (
        "明显偏离已发目标"
        if detail["stop_reason"] == "large_deviation"
        else f"连续超差 {detail['outside_for_s']:.3f} 秒"
    )
    axes = []
    for axis in detail["exceeded_axes"]:
        factor, unit, name = (1000, "mm", "夹爪") if axis == 6 else (1, "rad", f"J{axis + 1}")
        axes.append(
            f"{name}：已发目标 {detail['reference_target'][axis] * factor:.4f} {unit}，"
            f"反馈 {detail['feedback'][axis] * factor:.4f} {unit}，"
            f"误差 {detail['absolute_error'][axis] * factor:.4f} {unit}"
            f"（阈值 {TRACK_TOLERANCE[axis] * factor:.4f} {unit}）"
        )
    return (
        f"{stage} {detail['elapsed_s']:.3f} 秒时从臂跟随误差过大，已停止：{reason}；"
        + "；".join(axes)
        + f"。参考已发目标距今 {detail['reference_age_s'] * 1000:.0f} ms"
    )


class RobotReplay:
    def __init__(self, plan, channel, *, mock=False, factory=None):
        self.plan, self.channel = plan, channel
        self.factory = factory or (MockReplayTransport if mock else SocketReplayTransport)
        self.id = secrets.token_hex(16)
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.heartbeat_at = time.monotonic()
        self._status = {
            "id": self.id,
            "active": True,
            "phase": "checking",
            "elapsed_s": 0,
            "duration_s": float(plan.times[-1] + END_HOLD_S),
            "start_elapsed_s": 0,
            "start_duration_s": None,
            "error": "",
            "mock": mock,
            "field": plan.field,
            "speed": plan.speed,
            "with_gripper": plan.with_gripper,
            "tracking": None,
        }
        self.thread = threading.Thread(target=self._run, name="robot-replay", daemon=True)

    def status(self):
        with self.lock:
            return dict(self._status)

    def _update(self, **kwargs):
        with self.lock:
            self._status.update(kwargs)

    def heartbeat(self):
        with self.lock:
            self.heartbeat_at = time.monotonic()

    def _check_operator(self):
        if self.cancel.is_set():
            raise CaptureError("操作员停止回放")
        with self.lock:
            expired = time.monotonic() - self.heartbeat_at > 1.5
        if expired:
            raise CaptureError("浏览器心跳中断，回放已停止")

    def _run(self):
        transport, engaged = None, False
        try:
            transport = self.factory(self.channel, self.plan.points[0], self.plan.with_gripper)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                self._check_operator()
                transport.poll()
                self.cancel.wait(0.01)
            measured = transport.feedback()
            dimensions = 7 if self.plan.with_gripper else 6

            def exceeds(pose, target, tolerance):
                return np.any(np.abs(pose[:dimensions] - target[:dimensions]) > tolerance[:dimensions])

            start_pose = np.asarray(measured, dtype=np.float64).copy()
            start_duration = self.plan.start_duration(start_pose)
            self._check_operator()
            engaged = True
            transport.begin(measured)
            self._update(phase="moving_to_start", start_duration_s=start_duration + START_SETTLE_S)
            started = previous = time.monotonic()
            tracking = TrackingGuard(self.plan.with_gripper)
            tracking.sent(measured, started)
            play_started = settled_since = None
            while True:
                self._check_operator()
                now = time.monotonic()
                if now - previous > 0.15:
                    raise CaptureError("回放调度超时，已停止")
                previous = now
                measured = transport.feedback()
                encode_positions(measured, self.plan.with_gripper, context="实时反馈")
                now = time.monotonic()
                if now - previous > 0.15:
                    raise CaptureError("回放反馈检查超时，已停止")
                returning = play_started is None
                elapsed = now - (started if returning else play_started)
                detail = tracking.check(measured, now)
                detail.update(stage="returning" if returning else "playing", elapsed_s=float(elapsed))
                progress = {"start_elapsed_s" if returning else "elapsed_s": float(elapsed)}
                self._update(tracking=detail, **progress)
                if detail["stop_reason"]:
                    raise CaptureError(tracking_error(detail))
                if returning:
                    fraction = min(elapsed / start_duration, 1.0)
                    blend = fraction * fraction * (3 - 2 * fraction)
                    target = (
                        self.plan.points[0].copy()
                        if fraction == 1.0
                        else start_pose + (self.plan.points[0] - start_pose) * blend
                    )
                    if not self.plan.with_gripper:
                        target[6] = start_pose[6]
                else:
                    target = self.plan.at(elapsed)
                transport.target(target)
                tracking.sent(target, time.monotonic())
                if returning:
                    self._update(
                        phase="moving_to_start" if elapsed < start_duration else "settling_start",
                        start_elapsed_s=float(elapsed),
                    )
                    if elapsed >= start_duration:
                        if exceeds(measured, self.plan.points[0], START_TOLERANCE):
                            settled_since = None
                        elif settled_since is None:
                            settled_since = now
                        if settled_since is not None and now - settled_since >= START_SETTLE_S:
                            play_started = now
                            self._update(phase="playing", start_duration_s=float(elapsed))
                        elif elapsed >= start_duration + START_TIMEOUT_S:
                            raise CaptureError("回起点超时：从臂未稳定到达首帧位置，尚未开始轨迹回放")
                if not returning and elapsed >= self.plan.times[-1] + END_HOLD_S:
                    if exceeds(measured, target, START_TOLERANCE):
                        raise CaptureError("从臂未到达终点，已停止")
                    break
                self.cancel.wait(0.02)
            # Successful completion holds the final target without resetting the arm.
            self._update(phase="completed")
        except Exception as exc:
            self._update(phase="stopped", error=str(exc))
            if engaged and transport is not None:
                try:
                    transport.halt()
                except Exception as halt_error:
                    self._update(
                        phase="error", error=f"{exc}; 停止指令发送失败：{halt_error}，请使用硬件急停"
                    )
        finally:
            if transport is not None:
                try:
                    transport.close()
                except Exception as exc:
                    self._update(phase="error", error=str(exc))
            self._update(active=False)
            status = self.status()
            if status["error"]:
                LOGGER.warning("Replay stopped: %s", json.dumps(status, ensure_ascii=False))

    def close(self):
        self.cancel.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
