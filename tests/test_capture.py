import struct
from dataclasses import replace

import can
import numpy as np
import pytest

from piper_capture.config import ArmConfig, CameraConfig, CaptureConfig
from piper_capture.sources import SdkDecoder
from piper_capture.timeline import Aligner, CaptureError, Sample, Timeline


def config(tmp_path):
    return CaptureConfig(
        output=str(tmp_path / "dataset"),
        task="Pick the red book",
        fps=10,
        cameras=(CameraConfig("top_head", backend="opencv", width=48, height=32),),
        min_episode_frames=2,
        min_free_gb=0.001,
    ).validate()


@pytest.mark.parametrize(
    "identifier,expected_field,group",
    [
        (0x2A5, "state", "12"),
        (0x2A6, "state", "34"),
        (0x2A7, "state", "56"),
        (0x155, "action", "12"),
        (0x156, "action", "34"),
        (0x157, "action", "56"),
    ],
)
def test_real_sdk_decodes_signed_millidegrees(identifier, expected_field, group):
    message = can.Message(
        arbitration_id=identifier, data=struct.pack(">ii", 90_000, -45_000), is_extended_id=False
    )
    field, actual_group, values = SdkDecoder().decode(message)
    assert (field, actual_group) == (expected_field, group)
    np.testing.assert_allclose(values, [np.pi / 2, -np.pi / 4])


@pytest.mark.parametrize("identifier,field", [(0x2A8, "state"), (0x159, "action")])
def test_real_sdk_gripper_is_meters(identifier, field):
    message = can.Message(
        arbitration_id=identifier, data=struct.pack(">ihBB", 70_000, 1000, 1, 0), is_extended_id=False
    )
    actual_field, group, values = SdkDecoder().decode(message)
    assert (actual_field, group) == (field, "gripper")
    np.testing.assert_allclose(values, [0.07])


def test_malformed_can_is_rejected():
    with pytest.raises(CaptureError, match="payload"):
        SdkDecoder().decode(can.Message(arbitration_id=0x155, data=b"\x00", is_extended_id=False))


def test_interpolation_hold_and_stale_data():
    stream = Timeline(10)
    stream.append(Sample(1.0, np.array([1.0], np.float32)))
    stream.append(Sample(1.1, np.array([3.0], np.float32)))
    np.testing.assert_allclose(stream.vector(1.05, 0.2)[0], [2])
    np.testing.assert_allclose(stream.vector(1.05, 0.2, hold=True)[0], [1])
    with pytest.raises(CaptureError, match="gap"):
        stream.vector(1.05, 0.05)
    with pytest.raises(CaptureError, match="stale"):
        stream.vector(1.3, 0.05, hold=True)
    with pytest.raises(CaptureError, match="bracket"):
        stream.vector(0.5, 0.2)
    with pytest.raises(CaptureError, match="increase"):
        stream.append(Sample(1.0, np.zeros(1)))


def test_aligner_keeps_commands_separate_and_detects_missing_joint_pair(tmp_path):
    cfg = config(tmp_path)
    vectors = {}
    for field, value in (("state", 1.0), ("action", 2.0)):
        for group in ("12", "34", "56", "gripper"):
            stream = Timeline(10)
            for t in (1.0, 1.04):
                stream.append(Sample(t, np.full(1 if group == "gripper" else 2, value, np.float32)))
            vectors[f"right.{field}.{group}"] = stream
    images = Timeline(10)
    images.append(Sample(1.02, np.zeros((32, 48, 3), np.uint8)))
    aligner = Aligner(cfg, vectors, {"top_head": images})
    frame = aligner.frame(1.02)
    np.testing.assert_array_equal(frame.state, np.ones(7))
    np.testing.assert_array_equal(frame.action, np.full(7, 2))
    vectors["right.state.34"] = Timeline(10)
    with pytest.raises(CaptureError, match="right.state.34"):
        aligner.frame(1.02)


def test_config_rejects_ambiguous_sources_and_invalid_timing(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        replace(cfg, arms=(ArmConfig("left"), ArmConfig("right"))).validate()
    with pytest.raises(ValueError, match="same physical"):
        replace(cfg, cameras=(cfg.cameras[0], replace(cfg.cameras[0], name="hand_right"))).validate()
    with pytest.raises(ValueError, match="finite"):
        replace(cfg, alignment_delay_s=float("nan")).validate()


def test_command_hold_uses_previous_command_and_never_future_or_follower(tmp_path):
    cfg = config(tmp_path)
    vectors = {}
    for field in ("state", "action"):
        for group in ("12", "34", "56", "gripper"):
            stream = Timeline(10)
            size = 1 if group == "gripper" else 2
            times = (10.0, 10.04) if field == "state" else (1.0, 10.03)
            for i, t in enumerate(times):
                stream.append(Sample(t, np.full(size, 1 if field == "state" else 5 + i, np.float32)))
            vectors[f"right.{field}.{group}"] = stream
    images = Timeline(10)
    images.append(Sample(10.02, np.zeros((32, 48, 3), np.uint8)))
    aligner = Aligner(cfg, vectors, {"top_head": images})
    frame = aligner.frame(10.02)
    np.testing.assert_array_equal(frame.action, np.full(7, 5))
    np.testing.assert_array_equal(frame.state, np.ones(7))
    assert frame.timing["vectors"]["right.action.12"] == [1.0, 1.0]
    with pytest.raises(CaptureError):
        aligner.frame(11.0)
    vectors["right.action.12"] = Timeline(10)
    with pytest.raises(CaptureError, match="right.action.12"):
        aligner.frame(10.02)
