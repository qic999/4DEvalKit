# Full / Small 模型评测启动

`scripts/run_scannet_models.py` 将已有模型仓库中的 ScanNet 全量推理、3D 指标、框转换和 Qwen 问答串起来。**当前自动启动范围是 ScanNet val88 的全部 88 场景、每模型 2,071 道 VSI 问答。不是整个 VSI-Bench，也不是 12 项 core suite。**

这组实验明确使用 GT 2D proposals、GT category 和 GT camera poses；3D center / size / quaternion 来自重新运行的 encoder。Full 和 Small 使用相同场景、提示协议和 reasoning 参数。动态 benchmark 的带时间戳轨迹不能用这组静态合并框替代。

其他 benchmark 的 RGB → 预测检测框 → Full/Small → reasoning 入口见 [RGB evaluation](rgb_evaluation.md)。

复制 `configs/scannet_models.example.json` 为本机配置，填入 checkpoint、环境、数据和 GPU 路径。所有 GPU 必须互不重复；启动器在使用前检查显存占用。每个 encoder 的场景会按 clip 数分配到其 GPU。Qwen 在框生成之后启动，并由当前任务管理。

```bash
python -m scripts.run_scannet_models \
  --config configs/local_scannet_models.json \
  --output-root results/full_small_run1 --plan

bash scripts/run_background.sh logs/full_small_run1.log \
  python -u -m scripts.run_scannet_models \
  --config configs/local_scannet_models.json \
  --output-root results/full_small_run1
```

结果目录包含：

- `manifest.json`：配置、checkpoint 路径/大小/修改时间、QA hash、完整场景列表和输入协议。
- `status.json`：主 PID、每模型阶段、已完成场景数和失败原因，每 30 秒刷新。
- `runs/<variant>_scannet_val88_gt2d_all/logs/gpu_*.log`：实际 encoder worker 日志。
- `BoxDet/pred/<variant>_scannet_val88_gt2d_all/`：本次生成的逐帧框、相机预测、合并框和场景完成标记。
- `runs/<variant>_scannet_val88_gt2d_all/metrics.log`：原模型程序计算的 3D 框指标。
- `geometry/<variant>.json`：保留完整 quaternion 的几何，以及 GT / prediction 来源记录。
- `qa/<variant>.jsonl`：逐题输出，运行中持续保存；`qa/<variant>.json` 是完成后的评分结果。
- `logs/qwen_server.log`、`logs/<variant>_qa.log`：服务启动和问答进度。

```bash
cat results/full_small_run1/status.json
python -m scripts.model_status results/full_small_run1
tail -f logs/full_small_run1.log
```

若任务失败，先查看 `status.json` 指定的日志；修复环境后使用原配置、原输出目录和 `--resume`，并为后台启动日志选一个新名字。不要同时启动两个相同输出目录的任务。更换 checkpoint / 协议时必须使用新目录。

在 48 GB GPU 上，Full 的后置 detector segmentation 可能形成显存峰值。当前 3D 框评估只使用其 detection boxes / scores；这部分掩码在所有 tracker 帧计算完成后产生，不参与 3D 框导出或后续跟踪。可通过以下配置启用 `encoder_entry.py`，保留检测框回归、分类、候选置信度融合，跳过未使用的 detector mask 计算：

```json
{
  "encoder_python": "/path/to/4DEvalKit/scripts/encoder_python.sh",
  "encoder_base_python": "/path/to/sam3/bin/python",
  "box_only_detector_fusion": true
}
```

此选项只适用于当前原生 joint model 的 bbox 导出，不用于 segmentation 指标。2026-09-14 的运行在 3 个场景、两个模型的前 3 个 clip 上比较了 288 帧、2,052 个物体观测：3D box、quaternion、confidence 和 GT camera 数据完全一致；辅助 predicted camera pose 的最大数值差约 `3.04e-7`，该 GT-pose 问答实验不使用这一辅助位姿。运行目录保留了逐项比较和代码快照。

密集物体片段还可能在 tracker 的时序注意力阶段 OOM。设置 `FOURDEVAL_TRACKER_MASK_OFFLOAD=1` 可在每帧的跟踪、交互修正和 memory encoding 完成后，把该帧的分割 mask 输出复制到 CPU。时序 memory、memory selection、物体数量、帧数和 3D/pose 输出保留原逻辑。Full 同时需要上述 box-only detector fusion。该选项用于当前 bbox 推理接口，不适用于后续交互式 mask 编辑。

本次 `full_small_20260914_r2` 的恢复验证使用 Full 的 `scene0645_00`：零基 clip 340 的 16 帧、146 个物体观测中，3D box、quaternion、confidence、GT 和预测 camera pose 的最大绝对差均为 0；原先失败的 clip 341 随后成功导出 154 个观测。证据在运行目录 `recovery/mask_offload_probe/parity.json`。

`scripts/recover_scannet_workers.py --plan <worker_plan.json>` 接管明确列出的 worker PID / Linux start tick，保留正常 worker；遇到已记录的 OOM 时，用 mask offload 从失败 clip 重试一次。它不会缩减物体或跳过失败 clip；重试仍失败时记录错误。恢复状态独立保存在 `recovery/status.json`，`scripts.model_status` 会同时报告原 worker 和恢复进程。原始错误日志保留，恢复记录追加在同一日志中。

所有分片完成且原 controller / observer 退出后，恢复进程自动执行以下步骤，再生成汇总。此模式要求所有场景都有 `done.flag`，不启动 encoder，并复用已完成且覆盖正确的 QA：

```bash
python -m scripts.run_scannet_models --config configs/local_scannet_models.json \
  --output-root results/full_small_run1 --resume --postprocess-only
```

`max_existing_gpu_memory_mb` 默认 `1024`，可显式调整为共享机器允许的初始显存占用；它不锁定 GPU，也不保证其他任务之后不会占用显存。`data_prevalidated` 仅用于已经核对同一份数据且保留验证日志的恢复运行。

VSI 默认使用 PhysBrain scorer。若需和旧 `acc.py` 结果对比，可对新生成的 `qa/<variant>.json` 离线使用 `--vsi-metric-protocol project`，无需再次调用 Qwen。两种 overall 的定义不同，报告中应分列。

可在主任务启动后再启动汇总观察器。它检查每个模型是否生成 2,071 条唯一、有效的 completion，再对同一份输出重算项目评分，保存 `summary.json` 和 `summary.csv`。主任务失败时观察器退出并报告原因。

```bash
bash scripts/run_background.sh logs/full_small_run1.summary.log \
  python -u -m scripts.watch_scannet_results --run-dir results/full_small_run1
```

STI-Bench / VLM4D 及其余 core benchmark 目前仍由 `scripts/run_suite.py` 消费已导出的对应几何；该启动器不会将它们写成已排队或已完成。

## Elastic GPU scheduling

`scripts.run_elastic_scannet` supervises a shared queue of remaining Full scenes while adopting existing Small workers. Available GPUs load one Full checkpoint each and claim scenes through `scripts.elastic_queue`; a model stays loaded between jobs. GPUs released by Small automatically join the Full queue. GPU 7 participates in encoding and is released before the postprocessing launcher starts Qwen.

```bash
python -u -m scripts.run_elastic_scannet --queue /path/to/run/elastic/queue.json
python -m scripts.model_status /path/to/run
```

The queue records the original configuration, GPU pool, adopted PIDs and process start ticks, and scene jobs with `shard`, `scene_start`, `start`, and exclusive `stop` clip indices. Construct intervals from the original native shard manifests and resume each unfinished scene after its last successfully written clip. Stop the previous Full writers and retire the previous controller before assigning their remaining scenes; retain healthy adopted Small workers. Do not reindex clips or allow two workers to write the same scene.

Each worker validates scene boundaries against the native dataset. Scene completion is recorded only after the entire remaining interval returns successfully. Queue ownership uses a file lock and atomic writes. A failed scene remains visible while other scene jobs continue; logged CUDA OOMs receive one retry from the last successful clip. Runtime state, logs, queue and takeover records live under `elastic/`.

For dense DA3 inputs, `FOURDEVAL_LARGE_INTERPOLATION=1` splits bilinear interpolation along the independent object batch dimension when the input exceeds CUDA's `INT_MAX` element limit. No object or pixel is removed. The recovery check for `scene0645_00` reproduced all exported values exactly on clip 454 (16 frames, 82 observations) and successfully completed the previously failing clip 455. Evidence is saved in `elastic/interpolation_probe/parity.json`.
