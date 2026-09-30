import time
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from piper_capture.config import CameraConfig, CaptureConfig
from piper_capture.session import CaptureSession
from piper_capture.sources import Sources
from piper_capture.storage import DatasetStore, read_json
from piper_capture.timeline import CaptureError, Frame
from piper_capture.validate import validate_dataset


def config(tmp_path):
    return CaptureConfig(
        output=str(tmp_path / "dataset"),
        task="Pick the red book",
        fps=10,
        cameras=(CameraConfig("top_head", backend="opencv", width=48, height=32),),
        min_episode_frames=2,
        min_free_gb=0.001,
    ).validate()


def stage(store, task="Pick the red book", count=4):
    writer = store.start(task)
    for i in range(count):
        writer.append(
            Frame(
                100 + i / store.config.fps,
                np.full(7, i, np.float32),
                np.full(7, i + 1, np.float32),
                {"observation.images.top_head": np.full((32, 48, 3), i * 30, np.uint8)},
                {"cameras": {"top_head": {"sequence": i}}, "vectors": {}},
            )
        )
    return writer.finish()


def test_stream_save_resume_tasks_stats_and_video(tmp_path):
    cfg = config(tmp_path)
    with DatasetStore(cfg, mock=True) as store:
        assert store.accept(stage(store)) == 0
        first = (store.root / "data/chunk-000/episode_000000.parquet").read_bytes()
    with DatasetStore(cfg, resume=True, mock=True) as store:
        assert store.accept(stage(store, task="Pick the blue book")) == 1
        assert (store.root / "data/chunk-000/episode_000000.parquet").read_bytes() == first
    result = validate_dataset(Path(cfg.output))
    assert result["episodes"] == 2
    assert result["frames"] == 8
    assert not result["same_frame_action_equals_state_episodes"]
    stats = read_json(Path(cfg.output) / "meta/stats.json")
    np.testing.assert_allclose(stats["action"]["mean"], np.full(7, 2.5))
    np.testing.assert_allclose(stats["action"]["std"], np.full(7, np.std([1, 2, 3, 4])))
    assert stats["observation.images.top_head"]["count"] == [8]
    data = pq.read_table(Path(cfg.output) / "data/chunk-000/episode_000001.parquet").to_pydict()
    assert data["task_index"] == [1] * 4
    assert data["index"] == [4, 5, 6, 7]


def test_rejected_or_incomplete_episode_never_enters_training(tmp_path):
    cfg = config(tmp_path)
    with DatasetStore(cfg) as store:
        pending = stage(store, count=1)
        with pytest.raises(CaptureError, match="quality"):
            store.accept(pending)
        rejected = store.reject(pending)
        assert rejected.is_dir()
        assert read_json(store.root / "meta/info.json")["total_episodes"] == 0
        writer = store.start("broken")
        with pytest.raises(CaptureError, match="empty"):
            writer.finish()
        assert store.pending()[0]["issues"] == ["incomplete recording"]


def test_resume_rejects_wrong_contract_and_writer_lock(tmp_path):
    cfg = config(tmp_path)
    with DatasetStore(cfg):
        with pytest.raises(CaptureError, match="another process"):
            DatasetStore(cfg, resume=True)
    with pytest.raises(ValueError, match="contract differs"):
        DatasetStore(replace(cfg, fps=20), resume=True)
    with pytest.raises(ValueError, match="contract differs"):
        DatasetStore(cfg, resume=True, mock=True)
    with pytest.raises(FileExistsError):
        DatasetStore(cfg)


def test_interrupted_commit_recovers_without_duplicate_episode(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    with DatasetStore(cfg) as store:
        pending = stage(store)
        real_replace = __import__("os").replace
        failed = False

        def interrupted(src, dst):
            nonlocal failed
            if str(dst).endswith(".mp4") and not failed:
                failed = True
                raise OSError("simulated interrupted commit")
            return real_replace(src, dst)

        with monkeypatch.context() as patch:
            patch.setattr("piper_capture.storage.os.replace", interrupted)
            with pytest.raises(OSError, match="interrupted"):
                store.accept(pending)
    with DatasetStore(cfg, resume=True) as store:
        assert len(store.records()) == 1
        assert not store.pending()
    assert validate_dataset(Path(cfg.output))["episodes"] == 1


def test_tampered_video_is_rejected_before_commit(tmp_path):
    cfg = config(tmp_path)
    with DatasetStore(cfg) as store:
        pending = stage(store)
        (pending / "observation.images.top_head.mp4").write_bytes(b"broken")
        with pytest.raises(Exception):
            store.accept(pending)
        assert not store.records()
        assert not (store.internal / "commit.json").exists()


def test_interrupted_staging_cleanup_is_recoverable(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    with DatasetStore(cfg) as store:
        pending = stage(store)

        def interrupted_cleanup(path):
            raise OSError("cleanup interrupted")

        with monkeypatch.context() as patch:
            patch.setattr("piper_capture.storage.shutil.rmtree", interrupted_cleanup)
            with pytest.raises(OSError, match="cleanup interrupted"):
                store.accept(pending)
        assert (store.internal / "commit.json").exists()
    with DatasetStore(cfg, resume=True) as store:
        assert len(store.records()) == 1
        assert not store.pending()
    assert validate_dataset(Path(cfg.output))["episodes"] == 1


def test_simulated_capture_uses_real_timing_and_storage(tmp_path):
    cfg = config(tmp_path)
    with DatasetStore(cfg, mock=True) as store, Sources(cfg, mock=True) as sources:
        sources.wait_ready()
        session = CaptureSession(sources, store)
        session.start()
        deadline = time.monotonic() + 4
        while session.writer.count < 5 and time.monotonic() < deadline:
            session.tick()
            time.sleep(0.002)
        assert session.writer.count == 5
        store.accept(session.stop())
    assert validate_dataset(Path(cfg.output))["frames"] == 5


def test_selection_export_delete_reindexes_and_separates_failures(tmp_path):
    cfg = config(tmp_path)
    with DatasetStore(cfg, mock=True) as store:
        for task in ("red", "blue", "red", "green"):
            store.accept(stage(store, task=task))
        failure_id = store.save_failure(stage(store, task="failed blue"))
        selection = [
            {"kind": "saved", "id": "1"},
            {"kind": "saved", "id": "3"},
            {"kind": "failed", "id": failure_id},
        ]
        archive = store.export_selected(selection)
        exported = tmp_path / "exported"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(exported)
        assert validate_dataset(exported / "success")["episodes"] == 2
        table = pq.read_table(exported / "success/data/chunk-000/episode_000001.parquet").to_pydict()
        assert table["index"] == [4, 5, 6, 7]
        assert table["task_index"] == [1] * 4
        assert read_json(exported / "failed/episode_000000/manifest.json")["outcome"] == "failure"
        assert read_json(exported / "selection.json")["episodes"][-1]["source_id"] == failure_id
        assert len(store.records()) == 4
        store.delete_selected(selection)
        assert [r["task"] for r in store.records()] == ["red", "red"]
        assert not store.failed_records()
        assert validate_dataset(store.root)["episodes"] == 2
        assert read_json(store.root / "meta/info.json")["total_tasks"] == 1
        assert read_json(store.root / "meta/stats.json")["index"]["max"] == [7.0]
        store.delete_selected([{"kind": "saved", "id": "0"}, {"kind": "saved", "id": "1"}])
        assert validate_dataset(store.root)["episodes"] == 0
        assert store.accept(stage(store)) == 0
    with DatasetStore(cfg, mock=True, resume=True) as store:
        assert validate_dataset(store.root)["episodes"] == 1


def test_delete_recovers_after_directory_swap_interruption(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    with DatasetStore(cfg) as store:
        for task in ("red", "blue", "green"):
            store.accept(stage(store, task=task))
        real_replace = __import__("os").replace

        def interrupted(src, dst):
            if str(src).endswith("replacement/videos"):
                raise OSError("interrupted deletion")
            return real_replace(src, dst)

        with monkeypatch.context() as patch:
            patch.setattr("piper_capture.storage.os.replace", interrupted)
            with pytest.raises(OSError, match="interrupted deletion"):
                store.delete_selected([{"kind": "saved", "id": "1"}])
    with DatasetStore(cfg, resume=True) as store:
        assert [r["task"] for r in store.records()] == ["red", "green"]
        assert validate_dataset(store.root)["episodes"] == 2
        assert not (store.internal / "delete.json").exists()


def test_invalid_selection_cannot_delete_or_export(tmp_path):
    with DatasetStore(config(tmp_path)) as store:
        store.accept(stage(store))
        for selection in ([], [{"kind": "failed", "id": "../committed"}], [{"kind": "saved", "id": "0"}] * 2):
            for operation in (store.delete_selected, store.export_selected):
                with pytest.raises(ValueError):
                    operation(selection)
        assert validate_dataset(store.root)["episodes"] == 1
