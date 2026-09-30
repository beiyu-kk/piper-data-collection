import time
from collections import deque

import can
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from piper_capture.config import CaptureConfig
from piper_capture.replay import (
    END_HOLD_S,
    START_VELOCITY,
    TRACK_LAG_S,
    TRACK_PERSIST_S,
    MockReplayTransport,
    ReplayPlan,
    RobotReplay,
    SocketReplayTransport,
    TrackingGuard,
    encode_positions,
    tracking_error,
)
from piper_capture.sources import JOINT_SCALE, SdkDecoder
from piper_capture.timeline import CaptureError


def make_plan(tmp_path, points=None, *, speed=1.0, with_gripper=True, field="observation.state", **overrides):
    config = CaptureConfig(output=str(tmp_path), task="Replay")
    if points is None:
        points = [[0, 0.5, -0.5, 0, 0, 0, 0.04], [0.01, 0.5, -0.5, 0, 0, 0, 0.04]]
    path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table({field: points, "timestamp": np.arange(len(points), dtype=np.float32) / 30}), path
    )
    metadata = {"length": len(points), "issues": [], "contract": config.contract(), **overrides}
    return ReplayPlan.load(path, metadata, config, field, speed, with_gripper=with_gripper)


@pytest.mark.parametrize("speed", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("with_gripper", [False, True])
@pytest.mark.parametrize("field", ["observation.state", "action"])
def test_plan_preserves_recorded_timing_and_positions(tmp_path, speed, with_gripper, field):
    points = np.array(
        [
            [0, 0.5, -0.5, 0, 0, 0, 0.01],
            [1, 0.5, -0.5, 0, 0, 0, 0.10],
            [1, 0.5, -0.5, 0, 0, 0, 0.10],
            [-0.2, 0.5, -0.5, 0, 0, 0, 0.04],
        ]
    )
    plan = make_plan(tmp_path, points.tolist(), speed=speed, with_gripper=with_gripper, field=field)
    timestamps = pq.read_table(tmp_path / "episode.parquet")["timestamp"].to_numpy().astype(np.float64)
    np.testing.assert_array_equal(plan.points, points)
    np.testing.assert_array_equal(plan.times, timestamps / speed)
    for index, timestamp in enumerate(plan.times):
        np.testing.assert_array_equal(plan.at(timestamp), points[index])
    for index in range(len(points) - 1):
        midpoint = (plan.times[index] + plan.times[index + 1]) / 2
        np.testing.assert_allclose(plan.at(midpoint), (points[index] + points[index + 1]) / 2)
    description = plan.describe()
    assert description["timing_policy"] == "recorded_timestamps"
    assert description["trajectory_duration_s"] == pytest.approx(timestamps[-1] / speed)
    assert description["end_hold_s"] == END_HOLD_S
    assert description["duration_s"] == pytest.approx(timestamps[-1] / speed + END_HOLD_S)


def test_plan_rejects_invalid_values_and_quality(tmp_path):
    for invalid in (float("nan"), float("inf"), -float("inf"), 1e10):
        with pytest.raises(CaptureError):
            make_plan(tmp_path, [[invalid, 0.5, -0.5, 0, 0, 0, 0.04]] * 2)
    with pytest.raises(CaptureError, match="质量"):
        make_plan(tmp_path, issues=["corrupt"])


def test_tracking_matches_recent_sent_pose_and_expires_old_targets():
    guard = TrackingGuard(True)
    first, last = np.zeros(7), np.zeros(7)
    last[5], last[6] = 0.4, 0.02
    guard.sent(first, 0.0)
    guard.sent(last, 0.1)
    detail = guard.check(first, 0.12)
    assert detail["stop_reason"] is None and detail["exceeded_axes"] == []
    assert detail["reference_age_s"] == pytest.approx(0.12)
    assert detail["last_sent_target"] == last.tolist()
    assert detail["reference_target"] == first.tolist()
    detail = guard.check(first, TRACK_LAG_S + 0.01)
    assert detail["exceeded_axes"] == [5, 6]
    assert detail["stop_reason"] is None
    guard.sent(last, 0.36)
    detail = guard.check(first, TRACK_LAG_S + TRACK_PERSIST_S + 0.02)
    assert detail["stop_reason"] == "persistent_deviation"


@pytest.mark.parametrize("axis,gap", [(0, 0.3), (6, 0.02)])
def test_tracking_transient_error_recovers_and_resets_timer(axis, gap):
    guard = TrackingGuard(True)
    target, feedback = np.zeros(7), np.zeros(7)
    feedback[axis] = gap
    guard.sent(target, 0.0)
    assert guard.check(feedback, 0.01)["stop_reason"] is None
    guard.sent(target, 0.17)
    assert guard.check(feedback, 0.18)["stop_reason"] is None
    assert guard.check(target, 0.19)["outside_for_s"] == 0.0
    assert guard.check(feedback, 0.2)["outside_for_s"] == 0.0
    guard.sent(target, 0.38)
    assert guard.check(feedback, 0.39)["stop_reason"] is None
    detail = guard.check(feedback, 0.41)
    assert detail["stop_reason"] == "persistent_deviation"
    assert detail["exceeded_axes"] == [axis]
    detail.update(stage="playing", elapsed_s=1.234)
    message = tracking_error(detail)
    assert "1.234 秒" in message and "连续超差" in message
    assert "已发目标" in message and "反馈" in message
    assert ("J1" if axis == 0 else "夹爪") in message
    assert ("0.3000 rad" if axis == 0 else "20.0000 mm") in message


@pytest.mark.parametrize("axis,gap", [(0, 0.6), (6, 0.04)])
def test_tracking_large_deviation_stops_immediately(axis, gap):
    guard = TrackingGuard(True)
    target, feedback = np.zeros(7), np.zeros(7)
    feedback[axis] = gap
    guard.sent(target, 0)
    detail = guard.check(feedback, 0.01)
    assert detail["stop_reason"] == "large_deviation"
    assert detail["outside_for_s"] == 0
    assert detail["exceeded_axes"] == [axis]


def test_tracking_uses_one_complete_pose_and_ignores_disabled_gripper():
    guard = TrackingGuard(False)
    feedback = np.zeros(7)
    feedback[6] = 100
    guard.sent(np.zeros(7), 0.0)
    assert guard.check(feedback, 0.01)["stop_reason"] is None
    first, last = np.zeros(7), np.zeros(7)
    first[0], last[1] = 1, 1
    guard.sent(first, 0.2)
    guard.sent(last, 0.22)
    detail = guard.check(feedback, 0.23)
    assert detail["stop_reason"] == "large_deviation"
    assert 6 not in detail["exceeded_axes"]


def test_tracking_rejects_missing_or_expired_sent_history():
    guard = TrackingGuard(False)
    pose = np.zeros(7)
    with pytest.raises(CaptureError, match="发送中断"):
        guard.check(pose, 0.0)
    guard.sent(pose, 0.0)
    with pytest.raises(CaptureError, match="发送中断"):
        guard.check(pose, TRACK_LAG_S + 0.001)


class TrackingTransport(MockReplayTransport):
    def __init__(self, *args):
        super().__init__(*args)
        self.sent, self.halted, self.closed = [], False, False
        self.sent_times, self.sent_phases = [], []

    def target(self, pose):
        self.sent.append(pose.copy())
        self.sent_times.append(time.monotonic())
        self.sent_phases.append(self.replay.status()["phase"] if hasattr(self, "replay") else None)
        super().target(pose)

    def halt(self):
        self.halted = True

    def close(self):
        self.closed = True


def run_replay(plan, sink, *, heartbeat=True):
    replay = RobotReplay(plan, "fake", mock=True, factory=lambda *args: sink)
    sink.replay = replay
    replay.thread.start()
    deadline = time.monotonic() + 8
    while replay.thread.is_alive() and time.monotonic() < deadline:
        if heartbeat:
            replay.heartbeat()
        replay.thread.join(0.05)
    if replay.thread.is_alive():
        replay.close()
        pytest.fail("replay did not finish")
    return replay.status()


def test_replay_completes_and_holds_last_pose(tmp_path):
    points = np.tile([0, 0.5, -0.5, 0, 0, 0, 0.04], (31, 1))
    points[:, 0] = np.linspace(0, 0.8, len(points))
    points[:, 6] = np.linspace(0.04, 0.10, len(points))
    plan = make_plan(tmp_path, points.tolist())
    sink = TrackingTransport("fake", plan.points[0])
    status = run_replay(plan, sink)
    assert status["phase"] == "completed", status
    assert status["duration_s"] == 1.0 + END_HOLD_S
    assert status["duration_s"] <= status["elapsed_s"] < status["duration_s"] + 0.15
    np.testing.assert_allclose(sink.sent[-1], plan.points[-1])
    assert not sink.halted and sink.closed


def test_tracking_never_compares_feedback_with_unsent_future_target(tmp_path):
    points = [[0, 0.5, -0.5, 0, 0, 0, 0.04], [1, 0.5, -0.5, 0, 0, 0, 0.04]]
    plan = make_plan(tmp_path, points)
    sink = TrackingTransport("fake", plan.points[0])
    status = run_replay(plan, sink)
    assert status["phase"] == "completed", status
    assert not sink.halted
    np.testing.assert_array_equal(sink.sent[-1], plan.points[-1])


def test_original_speed_with_delayed_feedback_completes_without_stretching(tmp_path):
    points = np.tile([0, 0.5, -0.5, 0, 0, 0, 0.04], (61, 1))
    times = np.arange(len(points)) / 30
    points[:, 5] = 3.5 * np.minimum(np.minimum(times, 2 - times), 0.4)
    plan = make_plan(tmp_path, points.tolist())

    class Delayed(TrackingTransport):
        def __init__(self, *args):
            super().__init__(*args)
            self.pending = deque()
            self.max_latest_error = 0.0

        def target(self, pose):
            previous_feedback = self.pose.copy()
            super().target(pose)
            self.pending.append((time.monotonic(), pose.copy()))
            self.pose = previous_feedback

        def feedback(self):
            while self.pending and self.pending[0][0] <= time.monotonic() - 0.1:
                _, self.pose = self.pending.popleft()
            if self.sent:
                self.max_latest_error = max(self.max_latest_error, abs(self.sent[-1][5] - self.pose[5]))
            return self.pose.copy()

    sink = Delayed("fake", plan.points[0])
    status = run_replay(plan, sink)
    assert status["phase"] == "completed", status
    assert sink.max_latest_error > 0.25
    assert status["duration_s"] == 2.0 + END_HOLD_S
    assert status["duration_s"] <= status["elapsed_s"] < status["duration_s"] + 0.15
    np.testing.assert_array_equal(sink.pose, plan.points[-1])
    assert not sink.halted and sink.closed


def test_persistent_tracking_failure_has_snapshot_and_log(tmp_path, caplog):
    points = np.tile([0, 0.5, -0.5, 0, 0, 0, 0.04], (31, 1))
    points[:, 0] = np.minimum(np.arange(len(points)) * 0.05, 0.3)
    plan = make_plan(tmp_path, points.tolist())

    class Stalled(TrackingTransport):
        def feedback(self):
            return plan.points[0].copy()

    sink = Stalled("fake", plan.points[0])
    status = run_replay(plan, sink)
    assert status["phase"] == "stopped", status
    assert status["tracking"]["stop_reason"] == "persistent_deviation"
    assert status["tracking"]["outside_for_s"] >= TRACK_PERSIST_S
    assert status["tracking"]["exceeded_axes"] == [0]
    assert 0.4 < status["elapsed_s"] < 1.0
    assert status["tracking"]["elapsed_s"] == status["elapsed_s"]
    assert status["tracking"]["reference_target"][0] == pytest.approx(0.3)
    assert status["tracking"]["feedback"][0] == 0
    assert "J1" in status["error"] and "0.3000 rad" in status["error"]
    assert status["id"] in caplog.text and "persistent_deviation" in caplog.text
    assert sink.halted and sink.closed


def test_replay_heartbeat_and_tracking_fault_stop(tmp_path):
    plan = make_plan(tmp_path)
    sink = TrackingTransport("fake", plan.points[0])
    status = run_replay(plan, sink, heartbeat=False)
    assert "心跳" in status["error"] and sink.halted and sink.closed

    class BadTracking(TrackingTransport):
        def feedback(self):
            pose = super().feedback()
            if self.sent:
                pose[0] += 1
            return pose

    sink = BadTracking("fake", plan.points[0])
    status = run_replay(plan, sink)
    assert "误差" in status["error"] and sink.halted


def test_replay_precheck_never_transmits_on_invalid_feedback_or_conflict(tmp_path):
    plan = make_plan(tmp_path)
    invalid = plan.points[0].copy()
    invalid[0] = float("nan")
    sink = TrackingTransport("fake", invalid)
    status = run_replay(plan, sink)
    assert "数值无效" in status["error"] and not sink.sent and not sink.halted

    class Conflict(TrackingTransport):
        def poll(self):
            raise CaptureError("其他控制指令")

    sink = Conflict("fake", plan.points[0])
    status = run_replay(plan, sink)
    assert "控制指令" in status["error"] and not sink.sent and not sink.halted


@pytest.mark.parametrize("phase", ["moving_to_start", "playing"])
def test_replay_stop_during_return_or_playback(tmp_path, phase):
    plan = make_plan(tmp_path)
    sink = TrackingTransport("fake", plan.points[0])
    replay = RobotReplay(plan, "fake", mock=True, factory=lambda *args: sink)
    sink.replay = replay
    replay.thread.start()
    deadline = time.monotonic() + 5
    while replay.status()["phase"] != phase and time.monotonic() < deadline:
        replay.heartbeat()
        time.sleep(0.01)
    assert sink.sent and replay.status()["phase"] == phase
    replay.close()
    assert not replay.thread.is_alive() and sink.halted and sink.closed


def test_return_to_recorded_start_before_playing_at_bounded_speed(tmp_path):
    first = np.array([0, -0.0028274334, 0.0054628807, 0.095644, 0.197641, -0.124878, 0.1004])
    last = first.copy()
    last[0] = 0.01
    plan = make_plan(tmp_path, [first.tolist(), last.tolist()])
    measured = first.copy()
    measured[0] += 0.12
    measured[6] = 0.0716
    sink = TrackingTransport("fake", measured, True)
    status = run_replay(plan, sink)
    assert status["phase"] == "completed", status
    assert status["start_duration_s"] >= plan.start_duration(measured)
    assert "moving_to_start" in sink.sent_phases and "settling_start" in sink.sent_phases
    first_play = sink.sent_phases.index("playing")
    np.testing.assert_array_equal(sink.sent[0], measured)
    np.testing.assert_array_equal(sink.sent[first_play - 1], first)
    np.testing.assert_array_equal(sink.sent[-1], last)
    returning = np.array(sink.sent[:first_play])
    intervals = np.diff(sink.sent_times[:first_play])[:, None]
    # Allow one CAN unit plus small wall-clock sampling skew when comparing sent targets.
    slack = np.array([1e-4] * 6 + [1e-5])
    assert np.all(np.abs(np.diff(returning, axis=0)) <= intervals * START_VELOCITY + slack)
    assert not sink.halted and sink.closed


def test_return_duration_ignores_gripper_when_not_replayed(tmp_path):
    plan = make_plan(tmp_path)
    plan.with_gripper = False
    measured = plan.points[0].copy()
    measured[6] += 0.5
    assert plan.start_duration(measured) == 0.5
    sink = TrackingTransport("fake", measured)
    status = run_replay(plan, sink)
    assert status["phase"] == "completed", status
    first_play = sink.sent_phases.index("playing")
    assert all(pose[6] == measured[6] for pose in sink.sent[:first_play])


def test_return_that_never_reaches_start_stops_without_playing(tmp_path):
    plan = make_plan(tmp_path)
    measured = plan.points[0].copy()
    measured[0] += 0.12

    class Stalled(TrackingTransport):
        def feedback(self):
            return measured.copy()

    sink = Stalled("fake", measured)
    status = run_replay(plan, sink)
    assert "回起点超时" in status["error"]
    assert "playing" not in sink.sent_phases and status["elapsed_s"] == 0
    assert sink.halted and sink.closed


def test_sdk_transport_codec_competitor_feedback_and_no_init(monkeypatch):
    class Bus:
        def __init__(self):
            self.received, self.sent = deque(), []

        def recv(self, timeout):
            return self.received.popleft() if self.received else None

        def send(self, msg, timeout):
            self.sent.append(msg)

        def shutdown(self):
            pass

    bus = Bus()
    monkeypatch.setattr(can, "Bus", lambda **kwargs: bus)
    pose = np.array([0.1, -162 * JOINT_SCALE, 313 * JOINT_SCALE, -0.1, 0.2, -0.3, 0.1004])
    transport = SocketReplayTransport("fake", pose, with_gripper=True)
    assert not bus.sent
    transport.begin(pose)
    assert [m.arbitration_id for m in bus.sent] == [0x155, 0x156, 0x157, 0x159, 0x151]
    decoder = SdkDecoder()
    decoded = np.concatenate([decoder.decode(m)[2] for m in bus.sent[:4]])
    np.testing.assert_allclose(decoded, pose, atol=JOINT_SCALE)
    assert int.from_bytes(bus.sent[0].data[4:8], "big", signed=True) == -162
    assert int.from_bytes(bus.sent[1].data[:4], "big", signed=True) == 313
    assert int.from_bytes(bus.sent[3].data[:4], "big", signed=True) == 100400
    assert bus.sent[-1].data[3] == 0xAD
    invalid_pose = pose.copy()
    invalid_pose[6] = 1e10
    sent_count = len(bus.sent)
    with pytest.raises(CaptureError, match="CAN int32.*夹爪"):
        transport.target(invalid_pose)
    assert len(bus.sent) == sent_count

    def packet(identifier, data):
        return can.Message(arbitration_id=identifier, data=data, timestamp=time.time(), is_extended_id=False)

    for identifier, original in zip((0x2A5, 0x2A6, 0x2A7, 0x2A8), bus.sent[:4]):
        payload = bytearray(original.data)
        if identifier == 0x2A8:
            payload[6] = 0xC0
        bus.received.append(packet(identifier, payload))
    bus.received.append(packet(0x2A1, bytes(8)))
    for identifier in range(0x261, 0x267):
        bus.received.append(packet(identifier, bytes([0, 0, 0, 0, 0, 64, 0, 0])))
    np.testing.assert_allclose(transport.feedback(), pose, atol=JOINT_SCALE)
    sent_count = len(bus.sent)
    bus.received.append(packet(0x2A1, bytes([2, 1, 1, 1, 1, 0, 0, 0])))
    with pytest.raises(CaptureError, match="急停状态.*arm_status=0x01.*重新连接不会解除急停"):
        transport.feedback()
    assert len(bus.sent) == sent_count
    bus.received.append(packet(0x2A1, bytes(8)))
    np.testing.assert_allclose(transport.feedback(), pose, atol=JOINT_SCALE)
    transport.health[0x261] = (time.monotonic() - 1, True)
    with pytest.raises(CaptureError, match="状态反馈缺失或过期.*0x261"):
        transport.feedback()
    bus.received.append(packet(0x261, bytes([0, 0, 0, 0, 0, 0, 0, 0])))
    with pytest.raises(CaptureError, match="J1 未使能或驱动故障.*0x00"):
        transport.feedback()
    bus.received.append(packet(0x261, bytes([0, 0, 0, 0, 0, 64, 0, 0])))
    np.testing.assert_allclose(transport.feedback(), pose, atol=JOINT_SCALE)
    transport.health[0x2A8] = (time.monotonic(), False)
    with pytest.raises(CaptureError, match="未使能"):
        transport.feedback()
    transport.health[0x2A8] = (time.monotonic(), True)
    bus.received.append(packet(0x155, bytes(8)))
    with pytest.raises(CaptureError, match="控制指令"):
        transport.poll()
    transport.halt()
    assert bus.sent[-2].arbitration_id == 0x150 and bus.sent[-2].data[0] == 1
    assert int.from_bytes(bus.sent[-1].data[:4], "big", signed=True) == 100400
    transport.states["12"] = (time.monotonic() - 1, pose[:2])
    with pytest.raises(CaptureError, match="新鲜"):
        transport.feedback()
    transport.close()
    bus.sent.clear()
    transport = SocketReplayTransport("fake", pose, with_gripper=False)
    transport.begin(pose)
    transport.halt()
    assert all(message.arbitration_id != 0x159 for message in bus.sent)
    transport.close()


def test_replay_accepts_large_recorded_gripper_with_or_without_gripper_control(tmp_path):
    config = CaptureConfig(output=str(tmp_path), task="Replay")
    path = tmp_path / "episode.parquet"
    points = [[0, 0.5, -0.5, 0, 0, 0, 0.1027]] * 2
    pq.write_table(pa.table({"action": points, "timestamp": [0, 1 / 30]}), path)
    metadata = {"length": 2, "issues": [], "contract": config.contract()}
    for with_gripper in (False, True):
        plan = ReplayPlan.load(path, metadata, config, "action", 0.25, with_gripper=with_gripper)
        assert plan.with_gripper is with_gripper
        np.testing.assert_array_equal(plan.points, points)


def test_recorded_positions_ignore_sdk_ranges_without_opening_can(tmp_path, monkeypatch):
    from piper_sdk.piper_param import C_PiperParamManager

    def unexpected_call(*args, **kwargs):
        pytest.fail("Position preflight must not query SDK ranges or open CAN")

    monkeypatch.setattr(can, "Bus", unexpected_call)
    monkeypatch.setattr(C_PiperParamManager, "GetCurrentPiperParam", unexpected_call)
    config = CaptureConfig(output=str(tmp_path), task="Replay")
    path = tmp_path / "episode.parquet"
    points = np.array([[0, -0.0028274334, 0.0054628807, 0, 0, 0, 0.1004]] * 2, dtype=np.float32)
    pq.write_table(pa.table({"observation.state": points.tolist(), "timestamp": [0, 1 / 30]}), path)
    before = path.read_bytes()
    plan = ReplayPlan.load(
        path, {"length": 2, "contract": config.contract()}, config, "observation.state", 0.25, True
    )
    np.testing.assert_array_equal(plan.points, points)
    np.testing.assert_array_equal(plan.at(0.01), points[0])
    assert plan.describe()["position_policy"] == "recorded_without_clipping"
    assert path.read_bytes() == before


def test_encoding_preserves_signed_int32_values_and_rejects_overflow():
    pose = np.array([0, -0.0028274334, 0.0054628807, 0, 0, 0, 0.1027])
    np.testing.assert_array_equal(
        encode_positions(pose.astype(np.float32), True), [0, -162, 313, 0, 0, 0, 102700]
    )
    for axis in range(7):
        scale = JOINT_SCALE if axis < 6 else 1e-6
        for raw in (-(2**31), 2**31 - 1):
            candidate = pose.copy()
            candidate[axis] = raw * scale
            assert encode_positions(candidate, True)[axis] == raw
        for raw in (-(2**31) - 1, 2**31):
            candidate[axis] = raw * scale
            with pytest.raises(CaptureError, match="CAN int32"):
                encode_positions(candidate, True)
    pose[0] = np.finfo(np.float64).max
    with pytest.raises(CaptureError, match="CAN int32"):
        encode_positions(pose, True)


def test_raw_feedback_replay_completes_outside_sdk_default_ranges(tmp_path):
    plan = make_plan(tmp_path, [[0, -0.0028274334, 0.0054628807, 0, 0, 0, 0.1004]] * 2)
    measured = plan.points[0].copy()
    measured[1] -= 0.001
    sink = TrackingTransport("fake", measured)
    status = run_replay(plan, sink)
    assert status["phase"] == "completed", status
    np.testing.assert_array_equal(sink.sent[0], measured)
    np.testing.assert_array_equal(sink.sent[-1], plan.points[-1])
    assert not sink.halted and sink.closed
