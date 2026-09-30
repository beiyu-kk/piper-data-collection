# 旧数据动作时间偏移核对

日期：2026-09-16。范围为指定的 `openpi/openpi-piper` 当前工作副本、旧采集项目两个目录，以及 `recognition_book` 下四个未标注的 LeRobot v2.1 数据集。未修改旧采集代码、openpi 代码或原数据，也没有训练模型。

## 结论

旧数据采集和格式转换阶段没有把 action 向未来移一帧：同帧 `action` 是从臂反馈 `qpos` 的副本。当前 openpi-piper 的五个 Piper 训练配置设置 `action_delta_timestamps_start=1`，在加载动作块时读取下一帧开始的反馈轨迹；随后执行 DeltaActions。不能因为磁盘上的 action 与 state 相等，就认定训练加载器没有做时间偏移。

该结论确认当前实现行为，不追溯证明每个历史 checkpoint 当时用了同一版本和配置。官方默认从偏移 0 开始，这个可配置字段是 Piper 分支增加的。

## 源码链路

- `/home/ubun/project/piper/data_collection/piper_data_collection (2)/scripts/collection/collect_single_arm.py:894` 取当前从臂 qpos；RGB 和 RGBD 分支分别在第 917、929 行设置 `action=qpos.copy()`。
- 重构目录 `/home/ubun/project/piper/data_collection/piper_data_collection/src/piper_data_collection/collection/single_arm.py:78` 读取从臂反馈；第 119、131 行同样复制 qpos。
- 旧转换器 `scripts/convert_piper_to_lerobot.py:427` 使用同一个 `index` 写 `arrays.state[index]` 和 `arrays.action[index]`；重构版 `conversion/pume.py:86` 也使用同一个 frame_idx。
- `openpi-piper/src/openpi/training/config.py` 第 1061、1088、1113、1145、1170 行设置 offset=1，对应 four_book 全参、book_color 全参、book_color 名为 lora 的配置，以及 block_direction 两种配置。
- `openpi-piper/src/openpi/training/data_loader.py:143` 使用 `[offset/fps, ..., (offset+action_horizon-1)/fps]` 取动作块。
- `openpi-piper/src/openpi/transforms.py:219` 从动作块中每个目标减去当前 state；上述五个配置启用了 7 维 delta，包括夹爪。
- `scripts/train_piper.py`、`scripts/train_piper_recognize_four_book.py` 的 main 保留所选配置的 data。`scripts/compute_norm_stats.py` 和 `compute_norm_stats_local_assets.py` 也使用同一个 create_torch_dataset 及 data_transforms；按当前实现重新计算统计时同样包含移帧和 delta。

## 实际数据检查

对以下四个数据集的全部 parquet 读取 state/action 进行逐元素相等检查：

| 数据集 | 轨迹数 | 帧数 | action == 同帧 state 的帧数 |
| --- | ---: | ---: | ---: |
| recognize_four_books_lerobot_v2.1 | 96 | 69,245 | 69,245 |
| recognize_book_color_lerobot_v2.1 | 40 | 24,320 | 24,320 |
| recognize_red_label_book_lerobot_v2.1 | 80 | 42,643 | 42,643 |
| recognize_black_label_book_0911_lerobot_v2.1 | 40 | 20,361 | 20,361 |
| 合计 | 256 | 156,569 | 156,569 |

另外对四个对应原始目录的 `episode_0.hdf5` 和转换后的 `episode_000000.parquet` 检查：HDF5 内 action 与 observations/qpos 相等；转换后的两组数组分别与原始数组 float32 值逐元素一致，帧数不变。支持“转换阶段没有移帧”的判断。

## 加载器运行验证

使用 openpi-piper 自身的 Python 环境、LeRobotDataset、create_torch_dataset 和真实 data_transforms，在 book_color 第 0 条轨迹第 73 帧检查完整 30 步动作块。

直接加载原文件时遇到兼容问题：当前 datasets 库不能识别 parquet schema 中的 Hugging Face `List` 特征元数据，报 `Feature type 'List' not found`。为单独验证时间偏移，在临时目录导出同一条轨迹，去除 parquet schema metadata，保留全部列和数值不变，并核对 Arrow 表在忽略 metadata 后完全相等。视频链接至原视频，episode 元数据仅保留该条轨迹。验证完成后删除临时目录。未修改原文件，也未把临时兼容处理算作原数据直接加载成功。

| offset | 输入帧 | 第一目标帧 | 首个目标经过 7 维 delta 后 |
| --- | ---: | ---: | --- |
| 0 | 73 | 73 | `[0, 0, 0, 0, 0, 0, 0]` |
| 1 | 73 | 74 | `[0, 0.000296712, -0.007033706, -0.061540313, 0.026529014, 0.084700823, 0]` |

对整个 `(30, 7)` 动作块做了数值断言：offset=1 时，原始动作恰为 parquet 第 74 至 103 帧；变换后恰为这些位置分别减去第 73 帧 state。offset=0 时对应第 73 至 102 帧。关节单位为 rad，夹爪为 m。

对于反馈副本数据，offset=0 会让当前帧对应的首个 delta 恒为零；后续动作仍是未来位移，不能据此说整个动作块全零。offset=1 避开这个恒零首项，但机器人静止时未来位移当然仍可为零。

## 额外发现及边界

- 当前配置常量中的 four_book 和 book_color 数据集路径不含 `recognition_book`，检查时这些路径不存在。本次运行仅在内存中将 repo_id 指向用户指定的真实目录。
- 名为 `pi05_piper_recognize_book_color_full_finetune_all_delta` 的配置实际上引用 four_book 数据常量。是否需要修正应根据该实验真实意图确认；本次未改动。
- 当前代码能确认读取阶段有移帧；旧数据本身不能证明某次历史训练用了哪个配置。不能将当前代码行为直接套用到全部旧 checkpoint。
- 新采集项目以主臂目标命令为 action，接入脚本 `scripts/openpi_bridge.py:68` 使用 offset=0；这与旧数据的反馈轨迹监督含义不同。
