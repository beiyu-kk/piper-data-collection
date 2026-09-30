"""Run with the existing openpi Python environment; no changes to openpi are needed."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def build_config(args):
    from openpi.training import config as configs

    dataset = args.dataset.expanduser().resolve()
    contract = load_json(dataset / "meta/capture_contract.json")
    if contract["action_source"] != "leader_command" or contract["action_representation"] != "absolute":
        raise ValueError("expected absolute leader command labels")
    if contract["joint_unit"] != "radian" or contract["gripper_unit"] != "meter":
        raise ValueError("unexpected joint/gripper units")
    if contract["names"] != [f"right_joint_{i}" for i in range(7)]:
        raise ValueError("the existing openpi Piper policy accepts only the single right arm")
    if args.mode != "check" and contract["mock"] and not args.allow_mock:
        raise ValueError("simulation data is for validation; use --allow-mock only for a deliberate test")
    info = load_json(dataset / "meta/info.json")
    if info["codebase_version"] != "v2.1" or info["total_frames"] < 2:
        raise ValueError("a nonempty LeRobot v2.1 dataset is required")
    cameras = set(contract["cameras"])
    expected = {"observation.images.top_head", "observation.images.hand_right"}
    if cameras == expected:
        factory = configs.LeRobotPiperDataConfig
    elif cameras == expected | {"observation.images.book_side"}:
        factory = configs.LeRobotPiperBookSideDataConfig
    else:
        raise ValueError(f"camera keys do not match the existing openpi policy: {cameras}")
    base = configs.get_config(args.base_config)
    signature = json.dumps(
        {
            "dataset": str(dataset),
            "contract": contract,
            "frames": info["total_frames"],
            "episodes": info["total_episodes"],
            "horizon": base.model.action_horizon,
            "delta_gripper": args.delta_gripper,
        },
        sort_keys=True,
    )
    asset_id = "capture_" + hashlib.sha256(signature.encode()).hexdigest()[:12]
    return dataclasses.replace(
        base,
        name="piper_capture",
        exp_name=args.exp_name,
        data=factory(
            repo_id=str(dataset),
            assets=configs.AssetsConfig(asset_id=asset_id),
            base_config=configs.DataConfig(prompt_from_task=True, action_delta_timestamps_start=0),
            delta_gripper_action=args.delta_gripper,
            binarize_gripper_outputs=False,
        ),
        assets_base_dir=str(args.assets.resolve()),
        checkpoint_base_dir=str(args.checkpoints.resolve()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.train_steps,
        wandb_enabled=args.wandb,
        resume=args.resume,
        overwrite=False,
    )


def dataset_for_config(config):
    from openpi.training import data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed = data_loader.TransformedDataset(
        raw, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    )
    return data_config, raw, transformed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "stats", "train"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--openpi-root", type=Path, default=Path("/home/ubun/project/piper/openpi/openpi-piper")
    )
    parser.add_argument("--base-config", default="pi05_piper_recognize_book_color_lora_all_delta")
    parser.add_argument("--assets", type=Path, default=PROJECT / "work_dirs/openpi_assets")
    parser.add_argument("--checkpoints", type=Path, default=PROJECT / "work_dirs/checkpoints")
    parser.add_argument("--exp-name", default="piper_capture")
    parser.add_argument("--delta-gripper", action="store_true")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=30000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.train_steps) <= 0 or args.num_workers < 0:
        parser.error("batch size / train steps must be positive and num workers nonnegative")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("max frames must be at least 2")
    sys.path.insert(0, str(args.openpi_root.resolve() / "src"))
    sys.path.insert(0, str(args.openpi_root.resolve() / "packages/openpi-client/src"))
    logging.basicConfig(level=logging.INFO)
    config = build_config(args)
    data_config, raw, transformed = dataset_for_config(config)
    stats_path = config.assets_dirs / data_config.asset_id
    if args.mode == "check":
        checks = []
        for index in sorted({0, min(len(raw) - 1, 10), len(raw) - 1}):
            sample, result = raw[index], transformed[index]
            absolute = np.asarray(sample["action"])
            expected = absolute.copy()
            dims = 7 if args.delta_gripper else 6
            expected[:, :dims] -= np.asarray(sample["observation.state"])[None, :dims]
            np.testing.assert_allclose(result["actions"], expected, atol=1e-6)
            assert result["actions"].shape == (config.model.action_horizon, 7)
            assert isinstance(result["prompt"], str) and result["prompt"]
            checks.append(
                {
                    "index": index,
                    "state_shape": list(result["state"].shape),
                    "action_shape": list(result["actions"].shape),
                    "prompt": result["prompt"],
                    "images": {k: list(v.shape) for k, v in result["image"].items()},
                }
            )
        print(
            json.dumps(
                {
                    "frames": len(raw),
                    "action_start_offset": 0,
                    "checks": checks,
                    "norm_stats_path": str(stats_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.mode == "stats":
        from openpi.shared import normalize
        from tqdm import tqdm

        count = min(len(transformed), args.max_frames or len(transformed))
        stats = {key: normalize.RunningStats() for key in ("state", "actions")}
        # Include the final partial batch; never silently drop short episodes or the dataset tail.
        for start in tqdm(range(0, count, args.batch_size), desc="Normalization statistics"):
            batch = [transformed[i] for i in range(start, min(count, start + args.batch_size))]
            for key, accumulator in stats.items():
                accumulator.update(np.stack([np.asarray(sample[key]) for sample in batch]))
        normalize.save(stats_path, {key: value.get_statistics() for key, value in stats.items()})
        print(f"Normalization statistics: {stats_path} ({count} frames)")
    else:
        if data_config.norm_stats is None:
            raise ValueError(f"normalization statistics are missing; run stats first: {stats_path}")
        spec = importlib.util.spec_from_file_location(
            "piper_openpi_train", args.openpi_root / "scripts/train.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(config)


if __name__ == "__main__":
    main()
