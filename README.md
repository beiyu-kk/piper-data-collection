# Piper Capture

面向松灵 Piper 主从臂的本地数据采集流程，直接保存 LeRobot v2.1，用于当前 `openpi-piper` 的 pi 系列训练。

采集流程参考 EVA 的 **从臂反馈 / 主臂命令分开采集、固定时间轴对齐、后台保存和质量检查**。实现放在当前目录的 `src/piper_capture`，`EVA-CLIENT` 保留为参考项目。

## 当前实现

- 单右臂 7 维采集，两路 RGB；提供第三路 `book_side` 配置。
- 双臂 14 维采集配置，排列为左臂 7 维、右臂 7 维。当前 openpi 的 Piper adapter 只支持 7 维，训练接入脚本会拒绝双臂数据，避免误用。
- 采集模式在硬件主从模式下只监听 SocketCAN，使用指定 Piper SDK 的协议解析器。只有操作员在独立真机回放窗口预检并确认启动后，才发送回放控制帧；不自动使能、回零、重置或改变主从配置。
- 本地浏览器采集台：RealSense 或 V4L2 RGB 实时画面、主从臂数值、逐条录制、审核保存、丢弃、同步视频与曲线回放、历史记录和断点续采。
- 流式 H.264 编码、有界写入队列、逐条数据检查、可恢复的提交日志。
- 模拟模式不连接 CAN 或相机，经过相同的时间对齐和保存流程。

这版采集 RGB。深度、标定和软件主从转发尚未实现。当前两台 D435 的序列号已经通过设备枚举核对，第三台相机序列号只是待核对的配置值。

## 安装

在当前项目目录执行：

```bash
uv sync --extra hardware --extra dev
source .venv/bin/activate
piper-capture --help
```

`pyproject.toml` 使用本机 `../openpi/piper_sdk`，对应 `/home/ubun/project/piper/openpi/piper_sdk`。迁移到其他机器时调整 `[tool.uv.sources]`。采集环境与 openpi 训练环境独立，不需要在采集环境安装 JAX、PyTorch 或整个 LeRobot。

## 浏览器采集

在项目目录启动实机服务：

```bash
.venv/bin/piper-capture web --config configs/right_arm.yaml
```

在本机浏览器打开 **http://127.0.0.1:8080**。端口占用时通过 `--port 8081` 指定其他端口。服务只绑定本机地址，静态资源和图标均在本地，不依赖外部 CDN。

1. 查看双路相机和主从臂状态。左侧设备按钮可以查看每组 CAN 数据和相机的接收频率、数据年龄，以及重新连接设备。
2. 填写任务指令，点击“开始录制”，或在输入框外按 **空格**。再次按空格或点击“停止并保存”会校验并保存为成功示教；按 **L** 或点击“保存为失败”会结束并单独保存失败示教。长按不重复触发；输入框、回放弹窗和数据记录页不触发录制快捷键。
3. 相机画面与主从臂状态之间显示成功、失败的已保存条数。失败示教保留视频和状态数据，放在 `.piper/failed/<id>`，不计入成功 LeRobot 训练集；“丢弃”仍只保留诊断文件，不计为失败示教。
4. 批量审核：勾选“停止后加入待审核”，再用开始/停止按钮或空格连续采集。也可随时点“停止并审核”图标结束一条。待审核列表支持逐条勾选、全选、批量接受、保存失败、丢弃；操作前确认数量。质量异常的条目不会被批量接受，界面逐项显示错误，已成功处理的条目保留结果，其余继续待审核。视频审核支持双路同步、逐帧、拖动、倍速、循环、关节曲线，以及单条保存成功/失败。无有效帧的失败尝试保留计数和诊断信息，没有可播放轨迹。
5. 在“数据记录”中筛选成功/失败、搜索任务、逐条勾选、全选当前列表，或输入条数选择前 N 条。下载图标打开导出窗口：填写**本机绝对路径的新目录**，点击“导出到目录”；也可选择“下载 ZIP”。已有目录和当前数据集的父/子目录拒绝写入。导出完成后显示实际路径，ZIP 提供重试下载链接。删除图标经确认后永久删除所选记录及视频。录制与真机回放期间不能修改或导出数据。
6. 本机导出与 ZIP 使用同样的结构。`success/` 为重建编号、索引、任务和统计的 LeRobot v2.1 子集，沿用旧 **LeRobot** 数据的 `episode_000000.parquet` / `episode_000000.mp4` 命名；`failed/episode_000000/` 为独立失败记录，含同名 parquet、分相机 MP4 和 manifest，不自动混入训练集。列表保留“原始轨迹”，在“导出名称”列实时显示所选记录重排后的名称；成功记录按原始编号升序从零编号，失败记录按导出请求中的列表顺序独立从零编号，不按勾选先后排序。取消选择、全选或按条数选择均会更新编号；搜索/筛选隐藏的已选记录仍计入编号。此预览不修改源文件。`selection.json` 给出原记录编号/UUID 与导出名称的对应。此处不导出旧原始 `episode_0.hdf5` 格式。导出到其他硬盘使用真实文件复制，删除源数据不影响导出结果。

刷新或关闭浏览器页面不会停止后台录制；达到配置中的最长时长后自动停止并进入审核。正常关闭服务时，正在录制的轨迹保留为异常待审核数据，不自动写入训练集。

计数表示当前数据集已持久化的成功/失败记录，不包含待审核或已丢弃记录。保存 API 返回同一次提交后的计数和目录快照；浏览器按服务实例与递增版本更新状态，避免旧轮询结果覆盖刚保存的结果。保存因质量检查失败时显示原因并保留待审核，成功计数不会虚增；重启时从磁盘恢复计数。

### 真机回放单条轨迹

在数据记录中点“查看”（待审核列表可点“审核”），在视频窗口下方点“真机回放”。本功能仅支持当前配置的单个右从臂，使用已采集的弧度/米数据；模拟模式只调用模拟控制端。

1. 先看视频和轨迹，选择“主臂目标轨迹”（默认，重发所记录的动作目标）或“从臂反馈轨迹”（重现实际走过的位置），默认速度为“1 倍（采集原速）”，也可选择 0.25 / 0.5 倍慢放，再点“预检轨迹”。默认只回放六轴，不发送夹爪指令；需要夹爪参与时勾选“同时回放夹爪”。这些选项不修改训练数据里的 state/action。
2. 窗口显示首帧位置和实时从臂位置。**当前主从共 CAN 的跟随接线不能直接用于程序回放：必须断开主臂的 CAN 控制连接，保留电脑与右从臂的通信，关闭其他控制程序。** 依据为 SDK `asserts/double_piper.MD` 的 “Control slave arm” 说明。按设备操作规程完成接线与使能，并确保从当前位置回到首帧及后续回放的路径无障碍。
3. 确认工作区域、夹爪和被抓物安全，硬件急停可用，勾选两项确认后点“回起点并回放”。开始先监听 1 秒检查竞争控制帧、反馈、故障与六轴使能；随后从实时反馈位置自动低速移动到所选轨迹的第 0 帧，无需手动对齐起点。选从臂反馈时回到首帧 state，选主臂目标时回到首帧 action。勾选夹爪时同时恢复夹爪位置，否则不发送夹爪控制指令。预检计划 120 秒有效，记录重新编号或删除后须重新预检。
4. 使用 SDK 协议编码，经独立 SocketCAN 发送关节位置，以及勾选参与回放的夹爪位置，在原高跟随模式下执行；不切换到需要重置的位置速度模式。正式回放按记录的 `timestamp` 以约 50 Hz 插值发送，默认 1 倍，即 `播放时间 = (timestamp - 首帧 timestamp) / 倍率`。不再按关节 0.35 rad/s、夹爪 30 mm/s 拉长片段，也不按固定跳变阈值拒绝。夹爪控制力矩参数仍为 0.5 N·m。界面分别显示轨迹时长和额外的 0.5 秒终点保持；回起点另计。SDK 文档说明高跟随模式 `0xAD` 下速度百分比参数无效，轨迹节奏由发送目标的时间决定。按记录时间发送目标不等于保证物理跟随完全一致，设备自身动态和反馈检查仍然有效。
5. 点“停止回放”、关闭回放窗口、切换到后台或关闭发起回放的标签页会停止；浏览器心跳超过 1.5 秒未收到、CAN/反馈故障、跟随误差过大或线程调度延迟也会停止。停止时发送 SDK 快速急停帧；夹爪参与回放且反馈新鲜时，额外发送夹爪当前位置保持指令；正常完成则保持终点。**软件无法保证在断线/总线故障时送达停止帧，不能替代硬件急停。** 急停后请在设备端按规程恢复，程序不自动清除急停或失能放落机械臂。
6. 回放后，恢复硬件主从接线与跟随状态，再通过设备详情“重新连接”清空回放期间接收的目标缓存，才能继续采集。

回起点使用关节空间的三次平滑插值，起止速度为零，峰值速度不超过关节 0.175 rad/s、夹爪 15 mm/s；按各轴位移自动计算时长。到达目标后，反馈需连续 0.2 秒处于首帧误差范围内（关节 0.10 rad，启用夹爪时 5 mm），才启动轨迹计时；未到位会继续保持首帧目标，最多等待 3 秒后停止，不直接开始轨迹。网页分别显示“正在回起点”“确认起点到位”“回放中”。轨迹预计时长不包含回起点；启动后显示单独的回起点进度。两个阶段均可停止，使用同样的故障和跟随误差检查。程序不做环境碰撞规划，也不发送设备回零、重置或使能指令。

跟随检查仅使用最近 150 ms 内已经成功发送的目标，选择一组完整的关节/夹爪目标与当前反馈比较，不拿尚未发送的下一目标要求从臂到位。相对这段目标历史仍超出关节 0.25 rad 或启用夹爪时 15 mm，且连续 200 ms 未恢复才停止；明显偏离历史目标（关节 0.50 rad 或夹爪 30 mm）立即停止。历史匹配使用同一时刻的一组目标，不把各轴不同时刻的位置拼在一起。上述时间窗口是软件容忍策略，不是设备保证的响应时间，也不会改变轨迹速度。

跟随停止时，页面列出具体轴、参考的已发目标、实时反馈、误差、回放时刻与连续超差时间。`/api/status` 的 `replay.tracking` 和服务日志同时保存这些数值、最后已发目标及匹配目标的发送时间差；该时间差不等于测得的机械臂延迟。夹爪抓住物体后可能持续达不到命令开度，程序没有接触识别，仍按夹爪持续误差判断，不自动忽略。这种情况应根据具体夹爪数值判断，不能当成关节滞后或采集数据损坏。

设备状态检查分别报告急停状态（`arm_status=0x01`）、控制器异常、具体关节/夹爪未使能或故障、以及缺失/过期的状态帧。异常停止后的急停状态不会通过刷新网页、重启服务或“重新连接”解除。SDK 0.6.1 的 `EmergencyStop(0x02)` 与 `ResetPiper()` 都发送 `MotionCtrl_1(0x02, 0, 0)`，后者文档明确说明机械臂可能立即失电下落；这不是纯软件界面复位。须先在现场可靠支撑机械臂及夹持物，再按设备流程恢复，程序不自动发送该指令，也不自动重新使能。

回放采用原始记录的位置，**不再根据项目自定义范围或 SDK 默认软件范围拒绝、裁剪关节和夹爪目标**。不读取或改写 SDK 限位参数，不修改设备固件限制。YAML 不支持 `replay_gripper_max_m` 覆盖项，旧自定义配置中的该键需要删除。

预检和发送前保留数据有效性与 CAN 编码检查：7 维弧度/米数据必须有限，执行的各轴换算为关节 0.001 度、夹爪 1 微米后必须能用协议规定的有符号 int32 表示。整组目标通过编码范围检查后才发送首个控制帧，防止溢出、回绕或部分发包。正常单位换算会量化到 CAN 协议精度，parquet 的 state/action 保持不变。页面显示“原值回放”和“无软件位置限位”。

记录中的负 J2、正 J3 以及约 100.4 / 102.7 mm 的夹爪值不会再因 SDK 默认零位边界或 70 mm 上限被软件拦截，停止时的夹爪保持同样按实时反馈原值编码。通过预检不保证设备能够执行所有目标，硬件自身的使能、故障和行程约束仍然有效。停止、低速回起点、起点到位确认、跟随误差及通信检查继续保留。

预检覆盖轨迹数值、单位、时间和编码范围；开始后检查在线控制条件，不做环境碰撞规划。实现已通过模拟动作、SDK 编解码与浏览器测试；实机运动需要操作员按上述步骤在现场验证。

默认导出到 `/media/ubun/16T/Dataset/piper_eva_data/recognition/export_时间戳`，可在导出窗口修改。例如填写 `/media/ubun/16T/Dataset/piper_eva_data/recognition/reviewed_export_001` 后，导出结构如下：

```text
reviewed_export_001/
  selection.json
  success/
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.top_head/episode_000000.mp4
    videos/chunk-000/observation.images.hand_right/episode_000000.mp4
    meta/...
  failed/
    episode_000000/
      episode_000000.parquet
      observation.images.top_head/episode_000000.mp4
      observation.images.hand_right/episode_000000.mp4
      manifest.json
```

失败示教没有有效帧时，导出保留 manifest 与诊断文件，可能没有 parquet 或可解码视频。训练数据目录使用导出后的 `success/`。ZIP 保留在源数据集 `.piper/exports/` 以供重试；不需要的 ZIP 可在没有下载进行时清理该目录。

`web` 自动续采本程序创建的数据集，并检查保存契约、实机/模拟模式和进程锁。其他来源的已有目录会被拒绝，不会覆盖数据。请勿同时用终端采集、其他浏览器服务或相机预览程序占用同一套设备；多个浏览器标签页连接同一个服务则共享同一录制状态。

仅测试界面时使用模拟模式，它不会打开真实设备：

```bash
.venv/bin/piper-capture web --mock --output work_dirs/web_demo --port 8081
```

`web --mock` 不指定 `--output` 时保存到 `work_dirs/web_simulation`。采样频率、相机序列号、CAN 接口和时长上限在 YAML 中配置，修改后重启服务。页面中夹爪为便于观察显示为 mm，保存仍使用米；关节显示和保存均为弧度。

## 先验证软件流程

```bash
piper-capture doctor --mock
piper-capture record --mock --output work_dirs/demo --episodes 2 --duration 3
piper-capture validate work_dirs/demo
piper-capture replay work_dirs/demo --episode 0
```

回放需要图形桌面，空格暂停，`q` 退出。无图形桌面时 `validate` 会解码全部视频并检查帧数、分辨率、FPS 和时间戳。

终端 `record --mock` 不指定 `--output` 时写入 `work_dirs/simulation`。模拟和真实数据不能追加到同一个数据集。终端 `record` 对已存在的目录必须显式使用 `--resume`，程序不会覆盖已有数据。

## 实机采集

### 1. 检查设备与接线

```bash
piper-capture devices
piper-capture doctor --config configs/right_arm.yaml
piper-capture preview --config configs/right_arm.yaml
```

默认使用 `can0`，两台 D435 为：

| 训练图像名称 | 相机序列号 |
| --- | --- |
| `top_head` | `339322074804` |
| `hand_right` | `346522074547` |

设备序列号不能证明物理安装位置，请在预览中确认顶视和腕部画面对应正确。
`preview` 不保存数据，即使主臂命令尚未到达也能检查相机画面；`q` 退出。

默认主臂与从臂已通过硬件/CAN 配置完成跟随，同一总线上应能看到以下全部消息：

| 来源 | CAN ID | 数据用途 |
| --- | --- | --- |
| 从臂 | `0x2A5`、`0x2A6`、`0x2A7` | 六个关节实际位置 |
| 从臂 | `0x2A8` | 夹爪实际开度 |
| 主臂 | `0x155`、`0x156`、`0x157` | 六个关节控制目标 |
| 主臂 | `0x159` | 夹爪控制目标 |

如果反馈和主臂命令在不同接口，分别配置 `state_can` 和 `command_can`。采集程序只监听，接口分开时仍需外部系统完成实际跟随。默认 SDK 协议使用上述无偏移 CAN ID；使用了 ID 偏移的设备需要另行适配。

静止时不要求主臂持续发送目标帧。每组主臂目标收到一次后保持到下一次真实命令到达，关节和夹爪都使用目标时刻之前的最近命令；从臂反馈仍插值并检查新鲜度，相机仍检查时间偏差。CAN 断流超过 0.5 秒或从臂/相机缺帧会阻止采集。

程序刚启动、尚未收到首组目标时，只要 CAN 监听、从臂反馈和相机就绪，浏览器即可按空格开始；此时显示“等待首组主臂目标”，有效帧数为 0。收到全部四组真实目标后自动开始写入固定 FPS 数据。该等待阶段不会用零值、从臂反馈或未来命令补出 action；未收到任何目标时无法生成有效成功轨迹。等待超过最长时长自动停止并保留诊断文件。

同一 CAN 总线上没有独立的主臂在线心跳判据：主臂静止不发目标与单独拔掉主臂线、但从臂仍在发反馈，无法仅凭目标静默区分。程序显示“目标保持”而不据此声称检测到主臂独立在线。2026-09-16 静止无过滤监听实际读到约 200 Hz 从臂反馈，但没有 `0x155/0x156/0x157/0x159`；新行为已在此状态下检查启动就绪，在完整实机示教中仍需核对目标接收。

### 2. 录制与审核

```bash
piper-capture record --config configs/right_arm.yaml --preview
```

默认输出：

```text
/media/ubun/16T/Dataset/piper_data/recognition_book/piper_capture_right_v21
```

在终端输入命令并回车，或在预览窗口按对应单字母键：

| 命令 | 操作 |
| --- | --- |
| `r` | 开始一条轨迹 |
| `s` | 停止录制，进入待审核状态，显示帧数和质量问题 |
| `v` | 回放当前待审核轨迹，需要图形桌面 |
| `a` | 接受轨迹；完整解码校验通过后提交到 LeRobot |
| `d` | 丢弃当前/待审核轨迹，保留在诊断目录，不进入训练集 |
| `t Retrieve the blue book` | 在终端修改下一条轨迹的任务文本，只能在空闲状态使用 |
| `q` | 退出；当前轨迹保留为待审核，不自动加入训练集 |

`s` 后可以在另一个终端用下面的命令回放待审核轨迹，ID 来自终端输出：

```bash
piper-capture replay /path/to/dataset --pending RECORDING_ID
```

检查内容包括 CAN 流完整性、关节采样间隔、相机时间偏差、重复相机帧比例、最小轨迹长度、视频与表格帧数一致性、数据维度和有限数值。失败轨迹不能通过 `a` 强行加入训练集。

默认 30 FPS、至少 30 帧、最长 180 秒；达到最长时长后自动进入审核。相机录制在独立线程，磁盘编码使用有界队列，队列满时中止当前轨迹并保留诊断文件。

需要终端无人值守采集时可用 `--episodes N --duration SECONDS`；每条轨迹通过自动检查后直接保存，因此它不包含人工成功判定。正式人工示教建议使用交互方式。

### 3. 续采与恢复

```bash
piper-capture record --config configs/right_arm.yaml --resume --preview
piper-capture pending --config configs/right_arm.yaml
piper-capture pending --config configs/right_arm.yaml --accept RECORDING_ID
piper-capture pending --config configs/right_arm.yaml --discard RECORDING_ID
```

使用了 `--output` 时，上述命令也传相同目录。恢复模拟数据时同时传 `--mock`。

只有已经完成编码、拥有完整 manifest 且通过校验的待审核轨迹可以恢复接受。进程在录制中被强制终止的轨迹可能没有可解码视频，这类文件保留用于排查，可通过 `--discard` 移出待审核列表。提交过程中中断会在下次 `--resume` 时自动完成提交。

同一目录只允许一个写入进程。请在退出采集后启动训练或读取完整数据集；数据和多个元数据文件的提交不是面向并发训练读取的原子快照。

## 保存格式

```text
dataset/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.top_head/episode_000000.mp4
  videos/chunk-000/observation.images.hand_right/episode_000000.mp4
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    stats.json
    capture_contract.json
    capture/episode_000000.timing.jsonl
    capture/episode_000000.config.json
  .piper/
    contract.json
    committed/
    pending/
    rejected/
```

| 字段 | 含义 |
| --- | --- |
| `observation.state` | 从臂实际反馈 `[j1, j2, j3, j4, j5, j6, gripper]` |
| `action` | 同一时间轴上的主臂绝对控制目标，排列相同 |
| `observation.images.top_head` | 顶视 RGB 视频 |
| `observation.images.hand_right` | 右腕 RGB 视频 |
| `timestamp` | `frame_index / fps`，每条轨迹从 0 开始 |
| `task_index` | 指向 `meta/tasks.jsonl` 的任务文本 |

向量是 `float32`，关节单位为弧度，夹爪是 SDK 开度换算的米，不做二值化。采集时不计算 delta。LeRobot 表格只保存训练字段，设备来源时间、相机帧号和 RealSense 设备时间戳放在 `meta/capture` 旁路文件中。

对齐采用有界延迟窗口：从臂关节线性插值、从臂夹爪保持；主臂所有控制目标取目标时刻之前的最近真实命令，静止期间保持；RGB 选择最近帧。每条 `capture.json` 的 `action_sampling` 记录该采样规则，旧数据集契约仍可续采。默认等待 120 ms，以便取得目标时刻两侧的采样；这只增加录制处理延迟，不参与机械臂控制。CAN 使用内核接收时间转换到主机单调时钟；相机使用读取返回时的主机单调时间。RealSense 设备时钟另存，不跨设备直接比较。

主机接收时间不能消除 USB、曝光和相机驱动延迟。这版是软件时间对齐，尚未验证硬件同步或端到端相机延迟。正式批量采集前应通过实际运动画面和关节曲线核对对齐效果。

`.piper` 是本程序的恢复依据，续采需要保留。`meta/stats.json` 是 LeRobot 原始特征统计；openpi 使用经过 delta 转换后的独立归一化统计，不能互相代替。

## 接入 openpi

训练脚本复用 `/home/ubun/project/piper/openpi/openpi-piper` 的代码和 Python 环境。它在运行时创建配置，不修改该项目源码。默认使用已有 pi0.5 LoRA 配置的模型与优化器，但数据配置改成主臂命令标签、`action_delta_timestamps_start=0`、关节 delta、连续绝对夹爪。

```bash
OPENPI_PY=/home/ubun/project/piper/openpi/openpi-piper/.venv/bin/python
DATASET=/media/ubun/16T/Dataset/piper_data/recognition_book/piper_capture_right_v21

piper-capture validate "$DATASET"
"$OPENPI_PY" scripts/openpi_bridge.py check "$DATASET"
"$OPENPI_PY" scripts/openpi_bridge.py stats "$DATASET"
"$OPENPI_PY" scripts/openpi_bridge.py train "$DATASET" --exp-name book_001
```

`check` 验证真实 LeRobot 加载、相机映射、任务 prompt 和 action chunk/delta 转换，不加载模型权重。`stats` 使用 openpi 的 `RunningStats` 计算均值、标准差、q01、q99，包括最后一个不完整 batch。默认统计与 checkpoint 放在本项目 `work_dirs` 下，可用 `--assets`、`--checkpoints` 改为大容量磁盘。

通过 `--base-config` 选择其他现有模型配置；训练仍需该配置的预训练权重实际可用。使用 `--delta-gripper` 可以训练夹爪 delta，计算统计和训练时必须保持一致。归一化统计目录标识包含数据集、帧数、动作长度及夹爪表示，追加数据或切换表示后需要重新计算。

模拟数据可以运行 `check`。仅做统计流程测试时显式使用 `stats ... --allow-mock`；不要把它当成实际训练数据。本次没有启动模型训练。

## 测试与设计依据

```bash
pytest -q
ruff check src tests scripts
.venv/bin/playwright install chromium
.venv/bin/python scripts/verify_web.py
```

`verify_web.py` 自动启动独立模拟服务，通过浏览器执行录制、刷新、回放、保存、丢弃和重连，检查桌面/手机布局与图像像素，关闭测试服务后完整校验数据集。截图和数据保存在 `work_dirs/web_check_*`。使用 `--url http://127.0.0.1:8080` 只读检查运行中的实机界面，不开始录制或重新连接设备。

详细源码依据与设计取舍见 [docs/design.md](docs/design.md)。
本机运行结果与实机检查边界见 [docs/verification.md](docs/verification.md)。
