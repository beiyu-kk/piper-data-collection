import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from piper_capture.config import CameraConfig, CaptureConfig
from piper_capture.storage import DatasetStore
from piper_capture.timeline import CaptureError
from piper_capture.validate import validate_dataset
from piper_capture.web import create_app


@pytest.fixture
def config(tmp_path):
    return CaptureConfig(
        output=str(tmp_path / "dataset"),
        task="Pick the red book",
        fps=10,
        cameras=(CameraConfig("top_head", backend="opencv", width=96, height=64),),
        min_episode_frames=2,
        min_free_gb=0.001,
    ).validate()


def wait_status(client, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    status = {}
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if predicate(status):
            return status
        time.sleep(0.03)
    pytest.fail(f"Timed out waiting for capture status: {status}")


def command(client, name, **kwargs):
    token = client.get("/api/status").json()["token"]
    return client.post(
        "/api/command",
        headers={"X-Capture-Token": token},
        json={"command": name, **kwargs},
    )


def record(client):
    wait_status(client, lambda s: s.get("ready"))
    assert command(client, "start", task="Retrieve the blue book").status_code == 200
    wait_status(client, lambda s: s["frames"] >= 4)
    assert command(client, "stop").status_code == 200
    return wait_status(client, lambda s: bool(s["pending"]))["pending"][0]


def test_web_record_review_accept_and_discard(config):
    with TestClient(create_app(config, mock=True)) as client:
        wait_status(client, lambda s: s.get("ready"))
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/status", headers={"Host": "foreign.example"}).status_code == 403
        assert client.post("/api/command", json={"command": "start"}).status_code == 403
        assert command(client, "start", task=" ").status_code == 409
        assert command(client, "start", task="Retrieve the blue book").status_code == 200
        assert command(client, "start", task="duplicate").status_code == 409
        assert command(client, "reconnect").status_code == 409
        wait_status(client, lambda s: s["frames"] >= 4)
        assert client.get("/").status_code == 200
        assert client.get("/api/status").json()["state"] == "RECORDING"
        image = client.get("/api/cameras/top_head.jpg")
        assert image.status_code == 200 and image.content.startswith(b"\xff\xd8")
        assert command(client, "stop").status_code == 200
        pending = wait_status(client, lambda s: bool(s["pending"]))["pending"][0]
        assert not pending["issues"]
        review = client.get(f"/api/recordings/pending/{pending['id']}").json()
        assert review["task"] == "Retrieve the blue book"
        assert review["chart"]["state"] != review["chart"]["action"]
        video = client.get(review["videos"]["top_head"], headers={"Range": "bytes=0-63"})
        assert video.status_code == 206 and len(video.content) == 64
        assert command(client, "accept", recording_id=pending["id"]).status_code == 200
        status = wait_status(client, lambda s: len(s["history"]) == 1)
        assert not status["pending"]
        assert client.get("/api/recordings/saved/0").json()["length"] == pending["length"]
        assert client.get("/api/recordings/saved/0/video/top_head").status_code == 200
        assert client.get("/api/recordings/saved/999").status_code == 404
        assert client.get("/api/recordings/pending/not-an-id").status_code == 404
        second = record(client)
        assert command(client, "discard", recording_id=second["id"]).status_code == 200
        assert len(client.get("/api/status").json()["history"]) == 1
        assert client.get(f"/api/recordings/pending/{second['id']}").status_code == 404
        assert command(client, "reconnect").status_code == 200
        wait_status(client, lambda s: s.get("ready"))
    result = validate_dataset(Path(config.output))
    assert result["episodes"] == 1
    assert not result["same_frame_action_equals_state_episodes"]


def test_web_shutdown_retains_interrupted_episode_and_releases_lock(config):
    app = create_app(config, mock=True)
    with TestClient(app) as client:
        wait_status(client, lambda s: s.get("ready"))
        assert command(client, "start", task=config.task).status_code == 200
        wait_status(client, lambda s: s["frames"] >= 3)
    assert not app.state.runtime.thread.is_alive()
    assert not app.state.runtime.preview_thread.is_alive()
    with DatasetStore(config, resume=True, mock=True) as store:
        pending = store.pending()
        assert len(pending) == 1 and pending[0]["issues"]
        assert not store.records()
    with TestClient(create_app(config, mock=True, resume=True)) as client:
        status = wait_status(client, lambda s: s.get("state") == "REVIEW")
        identifier = status["pending"][0]["id"]
        assert command(client, "accept", recording_id=identifier).status_code == 409
        assert command(client, "discard", recording_id=identifier).status_code == 200


def test_web_source_failure_stops_recording_and_reconnects(config):
    app = create_app(config, mock=True)
    with TestClient(app) as client:
        wait_status(client, lambda s: s.get("ready"))
        assert command(client, "start", task=config.task).status_code == 200
        wait_status(client, lambda s: s["frames"] >= 3)
        app.state.runtime.sources.workers[0].error = CaptureError("camera disconnected")
        status = wait_status(client, lambda s: s.get("state") == "REVIEW")
        identifier = status["pending"][0]["id"]
        assert not status["ready"]
        assert any("camera disconnected" in issue for issue in status["pending"][0]["issues"])
        assert command(client, "accept", recording_id=identifier).status_code == 409
        assert command(client, "discard", recording_id=identifier).status_code == 200
        assert command(client, "start", task=config.task).status_code == 409
        assert command(client, "reconnect").status_code == 200
        wait_status(client, lambda s: s.get("ready"))


def test_web_auto_stops_at_episode_limit(config):
    config = replace(config, max_episode_s=0.5)
    with TestClient(create_app(config, mock=True)) as client:
        wait_status(client, lambda s: s.get("ready"))
        assert command(client, "start", task=config.task).status_code == 200
        status = wait_status(client, lambda s: s.get("state") == "REVIEW")
        assert status["pending"][0]["length"] == 5
        assert not status["pending"][0]["issues"]


@pytest.mark.parametrize("failed_source", ["can:can0", "camera:top_head", "simulation"])
def test_preview_readiness_is_independent_of_other_source_errors(config, failed_source):
    from piper_capture.sources import Worker

    config = replace(
        config,
        cameras=config.cameras
        + (CameraConfig("hand_right", backend="opencv", device=1, width=96, height=64),),
    )
    app = create_app(config, mock=True)
    with TestClient(app) as client:
        wait_status(client, lambda s: s.get("ready"))
        failure = Worker(failed_source)
        failure.error = CaptureError("device disconnected")
        app.state.runtime.sources.workers.append(failure)
        status = wait_status(client, lambda s: "device disconnected" in s.get("source_error", ""))
        assert not status["can_start"] and not status["ready"]
        assert status["streams"]["top_head"]["ready"] == (failed_source == "can:can0")
        assert status["streams"]["hand_right"]["ready"] == (failed_source != "simulation")
        assert status["streams"]["right.state.12"]["ready"] == (failed_source == "camera:top_head")
        if failed_source == "can:can0":
            for camera in config.cameras:
                image = client.get(f"/api/cameras/{camera.name}.jpg")
                assert image.status_code == 200 and image.content.startswith(b"\xff\xd8")
        assert command(client, "start", task=config.task).status_code == 409


def test_web_immediate_stop_exposes_empty_recording_for_discard(config):
    with TestClient(create_app(config, mock=True)) as client:
        wait_status(client, lambda s: s.get("ready"))
        assert command(client, "start", task=config.task).status_code == 200
        assert command(client, "stop").status_code == 409
        status = wait_status(client, lambda s: bool(s["pending"]))
        assert status["pending"][0]["issues"]
        assert command(client, "discard", recording_id=status["pending"][0]["id"]).status_code == 200


def test_web_startup_error_is_visible(config):
    with DatasetStore(config, mock=True):
        with TestClient(create_app(config, mock=True, resume=True)) as client:
            status = wait_status(client, lambda s: s.get("state") == "ERROR")
            assert "another process" in status["error"]
            assert not status["ready"]
            assert command(client, "start", task=config.task).status_code == 409


def test_stationary_leader_stays_ready_and_can_record(config):
    app = create_app(config, mock=True)
    with TestClient(app) as client:
        wait_status(client, lambda s: s.get("ready"))
        sources = app.state.runtime.sources
        for key, stream in sources.vectors.items():
            if ".action." in key:
                stream.append = lambda sample: None
        time.sleep(0.3)
        status = wait_status(client, lambda s: s.get("ready") and s["streams"]["right.action.12"]["held"])
        assert all(value is not None for value in status["positions"]["action"])
        assert command(client, "start", task=config.task).status_code == 200
        wait_status(client, lambda s: s["frames"] >= 5)
        assert command(client, "save_success").status_code == 200
        assert client.get("/api/status").json()["success_count"] == 1
        sources.workers[0].error = CaptureError("CAN disconnected")
        wait_status(client, lambda s: not s["ready"])
        assert command(client, "start", task=config.task).status_code == 409


def test_save_results_batch_export_delete_and_stale_selection(config, tmp_path):
    import io
    import zipfile

    with TestClient(create_app(config, mock=True)) as client:
        wait_status(client, lambda s: s.get("ready"))
        for outcome in ("success", "failure"):
            assert command(client, "start", task=config.task).status_code == 200
            wait_status(client, lambda s: s["frames"] >= 4)
            assert command(client, f"save_{outcome}").status_code == 200
        s = client.get("/api/status").json()
        assert (s["success_count"], s["failure_count"]) == (1, 1)
        assert not s["pending"]
        assert client.get(f"/api/recordings/failed/{s['history'][1]['id']}").status_code == 200
        selections = [{"kind": r["kind"], "id": r["id"]} for r in s["history"]]
        assert command(client, "delete", selections=selections, revision="old").status_code == 409
        result = command(client, "export", selections=selections, revision=s["catalog_revision"])
        assert result.status_code == 200
        response = client.get(result.json()["download"])
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            bundle.extractall(tmp_path / "export")
        assert validate_dataset(tmp_path / "export/success")["episodes"] == 1
        assert (
            command(client, "delete", selections=selections, revision=s["catalog_revision"]).status_code
            == 200
        )
        s = client.get("/api/status").json()
        assert (s["success_count"], s["failure_count"]) == (0, 0)
        assert validate_dataset(Path(config.output))["episodes"] == 0


def test_empty_failure_saved_but_short_success_requires_review(config):
    with TestClient(create_app(config, mock=True)) as client:
        wait_status(client, lambda s: s.get("ready"))
        assert command(client, "start", task=config.task).status_code == 200
        assert command(client, "save_failure").status_code == 200
        assert client.get("/api/status").json()["failure_count"] == 1
        assert command(client, "start", task=config.task).status_code == 200
        assert command(client, "save_success").status_code == 409
        status = wait_status(client, lambda s: bool(s["pending"]))
        assert status["success_count"] == 0
        assert command(client, "mark_failure", recording_id=status["pending"][0]["id"]).status_code == 200
        assert client.get("/api/status").json()["failure_count"] == 2


def test_can_start_before_first_command_without_fabricating_actions(config, monkeypatch):
    import numpy as np

    from piper_capture.sources import MockSource
    from piper_capture.timeline import Sample, Timeline

    original = MockSource.run
    blocked = {}

    def start_silent(source):
        for key, timeline in source.vectors.items():
            if ".action." in key:
                blocked[key] = timeline
                timeline.append = lambda sample: None
        original(source)

    monkeypatch.setattr(MockSource, "run", start_silent)
    app = create_app(config, mock=True)
    with TestClient(app) as client:
        status = wait_status(client, lambda s: s.get("can_start"))
        assert not status["ready"]
        assert command(client, "start", task=config.task).status_code == 200
        status = wait_status(client, lambda s: s.get("awaiting_action"))
        assert status["frames"] == 0
        assert status["positions"]["action"] == [None] * 7
        # Actual leader targets arrive once, then stay unchanged without further CAN messages.
        now = time.monotonic()
        for key, timeline in blocked.items():
            Timeline.append(
                timeline, Sample(now, np.full(1 if key.endswith("gripper") else 2, 0.25, np.float32))
            )
        status = wait_status(client, lambda s: s["frames"] >= 4)
        assert not status["awaiting_action"]
        assert command(client, "save_success").status_code == 200
    import pyarrow.parquet as pq

    data = pq.read_table(Path(config.output) / "data/chunk-000/episode_000000.parquet").to_pydict()
    np.testing.assert_array_equal(data["action"], np.full((len(data["action"]), 7), 0.25))


def test_batch_review_continues_capture_and_reports_partial_failure(config):
    from piper_capture.storage import read_json, write_json

    with TestClient(create_app(config, mock=True)) as client:
        for _ in range(3):
            record(client)
        status = client.get("/api/status").json()
        ids = [row["id"] for row in status["pending"]]
        assert len(ids) == 3 and status["success_count"] == 0
        broken = Path(config.output) / ".piper/pending" / ids[1] / "manifest.json"
        manifest = read_json(broken)
        manifest["issues"] = ["interrupted capture"]
        write_json(broken, manifest)
        assert (
            command(client, "review_batch", decision="accept", recording_ids=[ids[0], "stale"]).status_code
            == 409
        )
        assert len(client.get("/api/status").json()["pending"]) == 3
        response = command(client, "review_batch", decision="accept", recording_ids=ids).json()
        assert response["processed"] == 2 and response["remaining"] == 1
        assert [row["ok"] for row in response["results"]] == [True, False, True]
        assert response["status"]["success_count"] == 2
        assert response["status"]["sequence"] > status["sequence"]
        assert [r["id"] for r in response["status"]["pending"]] == [ids[1]]
        response = command(client, "review_batch", decision="mark_failure", recording_ids=[ids[1]]).json()
        assert response["status"]["failure_count"] == 1
        assert response["status"]["pending"] == []
    with TestClient(create_app(config, mock=True, resume=True)) as client:
        status = wait_status(client, lambda s: s.get("ready"))
        assert (status["success_count"], status["failure_count"]) == (2, 1)


def test_export_local_directory_owns_files_and_download_can_retry(config, tmp_path):
    import os

    app = create_app(config, mock=True)
    with TestClient(app) as client:
        first = record(client)
        assert command(client, "accept", recording_id=first["id"]).status_code == 200
        second = record(client)
        assert command(client, "mark_failure", recording_id=second["id"]).status_code == 200
        status = client.get("/api/status").json()
        selection = {
            "revision": status["catalog_revision"],
            "selections": [{"kind": r["kind"], "id": r["id"]} for r in status["history"]],
        }
        for destination in ("relative", config.output, str(Path(config.output) / "export"), str(tmp_path)):
            assert command(client, "export", destination=destination, **selection).status_code == 409
        output = tmp_path / "exports" / "selected"
        response = command(client, "export", destination=str(output), **selection)
        assert response.status_code == 200, response.text
        assert response.json()["destination"] == str(output)
        assert validate_dataset(output / "success")["episodes"] == 1
        assert (output / "failed/episode_000000/episode_000000.parquet").exists()
        video = "videos/chunk-000/observation.images.top_head/episode_000000.mp4"
        assert os.stat(output / "success" / video).st_ino != os.stat(Path(config.output) / video).st_ino
        assert command(client, "export", destination=str(output), **selection).status_code == 409
        archive = command(client, "export", **selection).json()["download"]
        assert client.get(archive, headers={"Range": "bytes=0-31"}).status_code == 206
        assert client.get(archive).status_code == 200
        assert client.get(archive).status_code == 200
        assert command(client, "delete", **selection).status_code == 200
        assert validate_dataset(output / "success")["episodes"] == 1


def test_replay_preparation_mutual_exclusion_stop_and_reconnect(config):
    with TestClient(create_app(config, mock=True)) as client:
        pending = record(client)
        assert command(client, "accept", recording_id=pending["id"]).status_code == 200
        assert command(client, "replay_start").status_code == 409
        assert (
            command(client, "replay_prepare", recording_id="0", recording_revision="stale").status_code == 409
        )
        revision = client.get("/api/recordings/saved/0").json()["recording_revision"]
        prepared = command(
            client, "replay_prepare", recording_id="0", field="observation.state", recording_revision=revision
        ).json()
        assert prepared["plan"]["frames"] >= 4
        assert prepared["plan"]["position_policy"] == "recorded_without_clipping"
        assert prepared["plan"]["start_mode"] == "move_to_first_pose"
        assert prepared["plan"]["speed"] == 1.0
        assert prepared["plan"]["timing_policy"] == "recorded_timestamps"
        assert prepared["plan"]["tracking_policy"] == "recent_sent_targets"
        assert prepared["plan"]["trajectory_duration_s"] == pytest.approx(
            (prepared["plan"]["frames"] - 1) / config.fps
        )
        assert prepared["plan"]["duration_s"] == pytest.approx(
            prepared["plan"]["trajectory_duration_s"] + prepared["plan"]["end_hold_s"]
        )
        plan = prepared["plan"]["plan_token"]
        assert command(client, "replay_start", plan_token=plan).status_code == 409
        start = command(
            client, "replay_start", plan_token=plan, leader_disconnected=True, workspace_clear=True
        ).json()
        assert start["replay_id"]
        assert command(client, "start", task=config.task).status_code == 409
        assert command(client, "reconnect").status_code == 409
        assert command(client, "replay_heartbeat", replay_id="wrong").status_code == 409
        assert command(client, "replay_stop", replay_id=start["replay_id"]).status_code == 200
        stopped = wait_status(client, lambda s: s.get("replay") and not s["replay"]["active"])
        assert stopped["replay"]["phase"] == "stopped"
        assert stopped["replay"]["field"] == "observation.state"
        assert stopped["replay"]["speed"] == 1.0
        assert command(client, "start", task=config.task).status_code == 409
        assert command(client, "reconnect").status_code == 200
        assert (
            command(
                client, "replay_start", plan_token=plan, leader_disconnected=True, workspace_clear=True
            ).status_code
            == 409
        )
        wait_status(client, lambda s: s.get("ready"))
