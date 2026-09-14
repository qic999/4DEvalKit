# 4DEvalKit

面向 `spatial_encoder_v2` / `spatial_encoder_v2_small` 的 **3D boxes / 4D tracks → 文本 LLM → benchmark scoring** 工具库。

目录结构参考 [PhysBrainEvalKit](https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit)，复用 PhysBrain 1.5 Table 4 的 28 项数据/评分适配器，另外加入 STI-Bench、VLM4D、DSI-Bench。上游代码固定在 `4b37ca2f184bd76b98827481a6739f89ea06d730`，来源和修改清单见 [upstream_manifest.json](docs/upstream_manifest.json)。

**适配器可用不等于所有任务都能仅靠物体框充分解决。** 主评测建议先做 12 项：BLINK、CV-Bench、3DSRBench、EmbSpatial-Bench、Q-Spatial-Bench、MindCube、MMSI-Bench、ViewSpatial-Bench、VSI-Bench、SAT、STI-Bench、VLM4D。指点、部件 affordance、机器人技能规划和生成轨迹作为扩展。逐项判断见 [benchmark_matrix.md](docs/benchmark_matrix.md)。

## 目录

```text
4DEvalKit/
├── eval.py                    # 统一入口
├── eval_*.py                  # 与上游类似的单 benchmark 入口
├── benchmark/
│   ├── loader.py              # 样本 ID、原生数据加载、分批准备
│   ├── dynamic.py             # STI / VLM4D / DSI 原生标注适配
│   ├── physbrain/             # 隔离保存的 28 项上游实现
│   ├── legacy/vsi_acc.py      # 项目原有 VSI scorer 的快照
│   └── project_vsi.py         # 旧 VSI 评分兼容层
├── core/                      # 几何校验、文本提示、HTTP 推理、恢复、结果 IO
├── scripts/                   # 格式转换、suite、后台启动、汇总、验证
├── configs/                   # Full / Small 共用协议模板
├── examples/                  # 人工构造的小样例，不是模型成绩
├── docs/                      # 适用范围、schema、评分协议、验证记录
└── tests/                     # 格式、时序、评分、API、断点续跑回归测试
```

本库读取两个 encoder 导出的框，不在这里加载 encoder 权重或重新运行检测。两个目录都包含不同配置，必须用实际 checkpoint/config 标识 Full / Small，不能仅凭目录名推断模型规模。LLM 使用 OpenAI-compatible `/v1/chat/completions` 接口；API 客户端只发送文本，不需要在此环境安装 PyTorch。已有 VLM / LLM 输出也可离线重评分。

## 环境和快速检查

Python 3.10+，建议独立环境。从本仓库根目录运行所有命令。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python eval.py --list-benchmarks
python -m pytest -q tests
```

数据和模型权重不放进 Git。可自行设置 `HF_HOME`、`HF_DATASETS_CACHE`；`--data` 接受原生本地数据或 HF dataset ID。不传 `--data` 时采用 registry 的公开源，可能下载大型数据集。只想验证流程时先用下列人工样例：

```bash
python eval_vlm4d.py --data examples/dynamic_qa.json \
  --geometry examples/dynamic_geometry.json --require-tracks \
  --dry-run --output results/example_check.json

python eval_vlm4d.py --data examples/dynamic_qa.json \
  --predictions examples/dynamic_predictions.json \
  --model synthetic_fixture --run-label synthetic_smoke \
  --output results/example_replay.json
```

已有输出不会被静默覆盖。正常评估加 `--resume` 可恢复；dry-run 请用新文件名。

若 reasoning 服务默认启用 thinking，请按服务支持情况显式设置，例如 `--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`，并对 Full / Small 保持一致。MCQ 默认 512 个输出 token；生成长技能序列/轨迹时需明确增加 `--max-tokens`。HTTP 请求失败会保存结果并以非零状态退出；截断输出单独标记，不从推理内容中凑答案。

## 静态场景：接入已有 VSI 管线

`--qa-json` 是 `--data` 的别名，`--bbox-json` 是 `--geometry` 的别名。兼容项目现有 `scene_name → [{label,bbox_3d:[xyz,size,roll,pitch,yaw]}]` JSON，包括内容实际为 JSON 字典的 `.jsonl`。

```bash
python eval_vsi_bench.py \
  --qa-json /path/to/vsibench_qa.json --dataset scannet \
  --bbox-json /path/to/predicted_scene_boxes.json \
  --units m --coordinate-frame world_z_up \
  --model qwen35-9b --base-url http://localhost:8000/v1 \
  --temperature 0 --seed 0 --run-label small_gt2d_pred3d \
  --output results/small/vsi_scannet.json --resume
```

先加 `--dry-run --output results/vsi_check.json` 检查全部选中问题是否能匹配几何。`scannet` 对当前项目 QA 文件是 88 scenes / 2,071 QA；不等同于完整 VSI-Bench。

已有推理输出直接评分，不联系模型：

```bash
python eval_vsi_bench.py --qa-json /path/to/vsibench_qa.json --dataset scannet \
  --predictions /path/to/existing_qwen_results.json \
  --vsi-metric-protocol project --model qwen35-9b \
  --output results/replay_project.json
```

`--vsi-metric-protocol project` 调用项目旧 `acc.py`，用于复核已有结果；默认 `physbrain` 调用上游评分。两者 MRA 阈值、答案解析、相对方向合并和 overall 加权不同，不能直接比较。已有文件名的 `qwen_only` 指 Qwen reasoning 路径；该文件的 prompt 包含框，不能当成 no-box baseline。

建议把 legacy 框转换成带来源记录的新格式：

```bash
python -m scripts.convert_geometry --format legacy \
  --input /path/to/scene_boxes.json --output data/geometry/small/vsi_bench.json \
  --units m --coordinate-frame world_z_up \
  --geometry-source predicted --encoder spatial_encoder_v2_small \
  --checkpoint /path/to/exact_checkpoint.pt \
  --proposal-source gt_2d --label-source gt_category --camera-pose-source ground_truth
```

上面的 GT 标志只是混合输入实验的示例，必须按实际管线填写。预测框、GT 2D proposal、GT 类别、GT 相机位姿应分别记录，不能合称 fully predicted。

若输入是 `merge_json.py` 的 10D 合并框，使用 `--format merged --quaternion-order xyzw`。单场景文件加 `--scene-id`；多场景目录加 `--merged-name exact_filename.json`。转换保留完整 quaternion，不再只取 yaw。具体见 [geometry_schema.md](docs/geometry_schema.md)。

## 新 benchmark 的样本 ID

先导出公开问题和媒体索引，再为相同输入生成 encoder 框：

```bash
python eval_mmsi_bench.py --data /path/to/MMSI-Bench \
  --export-manifest data/manifests/mmsi.json
```

manifest 包含稳定 ID（如 `mmsi_bench:0`）、原始 ID、问题和可取得的媒体引用，不导出 answer、thought、GT mask。几何推荐用稳定 ID 作键。VSI 也可用 `scene_name` 共用一个场景。ID 依赖固定数据版本/顺序，不能拿同一 ID 映射到另一个 split。嵌入式图像默认只描述尺寸/类型；准备 encoder 输入可使用 `--export-media-dir` 保存真实图像/帧，见 schema 文档。

## 动态场景：逐帧框与统一参考系

动态输入使用 `tracks[].observations[]`：稳定 track ID、秒制 timestamp、每帧 center / size / quaternion，以及 camera-to-world 序列。不能把同一物体所有帧平均成一个框。`--require-tracks` 可检查输入确实含轨迹；它不自动判断 tracking 是否正确。

```bash
python eval_sti_bench.py --data /path/to/STI-Bench/qa.parquet \
  --geometry data/geometry/small/sti_bench.json --require-tracks \
  --model qwen35-9b --base-url http://localhost:8000/v1 \
  --run-label small_pred4d --output results/small/sti.json --resume

python eval_vlm4d.py --split real_mc --geometry data/geometry/small/vlm4d.json \
  --require-tracks --model qwen35-9b --base-url http://localhost:8000/v1 \
  --output results/small/vlm4d_real.json --resume
```

VLM4D 的 `real_mc`、`synthetic_mc` 分开跑。DSI 使用 `--split std` 或 `--split all`；`all` 从原生目录加载 std / reverse / hflip / reverse_hflip 四组，并报告 robustness。每个增强必须使用对应增强视频重新得到的几何。不能把原视频的框不加变换复用于倒放/镜像。

SpatialEncoder 原始 `json/<clip>/*.jsonl` 可转换成 tracks：

```bash
python -m scripts.convert_geometry --format raw-tracks \
  --input /path/to/one_scene_output --scene-id sti_bench:0 \
  --timestamps /path/to/frame_seconds.json --metric-scale 2.5 \
  --poses /path/to/global_camera_poses.json --track-map /path/to/global_track_ids.json \
  --coordinate-frame world_z_up --geometry-source predicted \
  --encoder spatial_encoder_v2_small --checkpoint /path/to/checkpoint.pt \
  --proposal-source predicted_2d --label-source detector \
  --output data/geometry/small/one_dynamic_scene.json
```

`2.5` 来自当前模型输出尺度约定，换模型时必须核对，代码不默认猜测。`pred_camera_pose` 通常是 clip-local；必须先做跨 clip 对齐再提供 `--poses`。若刻意做 GT-pose 实验，改用 `--record-ground-truth-poses`，结果会记录这一来源。只有确实跨 clip 保持对象 ID 一致时才能用 `--ids-global` 代替 `--track-map`。转换器不提供自动 tracking / pose alignment。

STI/VLM4D/DSI 当前是明确要求选项字母的确定性 MCQ 协议。VLM4D 原论文的自由文本输出使用模型 judge 和人工核验，因此这里标记 `vlm4d_direct_choice_v1`，不声称与论文原分数直接同口径。

## Full / Small 与消融，共用协议后台跑

编辑 [configs/core_suite.example.json](configs/core_suite.example.json)，填写同一 reasoning LLM 和各 encoder 的几何路径。支持 `3d`、`2d`、`none` 表示；2D 实验必须有真实 `bbox_2d`，不能从 3D GT 偷算；GT 3D oracle 用独立 geometry 文件和 `geometry_source=oracle`。RGB / RGB+boxes 的 VLM baseline 可用外部管线生成，再通过 `--predictions` 重评分。

```bash
cp configs/core_suite.example.json configs/local_core_suite.json
# 编辑 local_core_suite.json，填入实际模型服务和已生成的几何文件。
python -m scripts.run_suite --config configs/local_core_suite.json --mode plan
python -m scripts.run_suite --config configs/local_core_suite.json \
  --mode check --output-dir results/preflight

bash scripts/run_background.sh logs/core_suite.log \
  .venv/bin/python -m scripts.run_suite --config configs/local_core_suite.json \
  --mode run --output-dir results/core_suite --resume

tail -f logs/core_suite.log
python -m scripts.summarize_results results/core_suite --output results/core_scores.csv
```

后台 launcher 使用 `nohup + setsid`，记录 PID 和日志。Suite 顺序跑 benchmark，每个 benchmark 内有界并发请求 LLM，失败记录到 `suite_index.json`；已完成 job 不会因另一 job 失败而丢失。不要同时启动两个写同一结果文件的 suite。

每次评估产出最终 `.json`、逐条 `.jsonl`、配置 `.manifest.json`。恢复时核对数据、geometry、模型、生成参数、prompt 和逐题指纹，成功响应直接复用；请求失败、缺失、截断响应会保留状态并参与原生评分，不从样本总数中删除。上游轨迹指标对无效轨迹采用单独 valid denominator，结果会同时暴露 coverage，详见 [metric_protocols.md](docs/metric_protocols.md)。汇总 CSV 不自动计算冒充 Table 4 的 28 项 overall。

## 验证与范围

自动测试覆盖 31 个 adapter 导入、完整旋转、动态位姿/尺度变换、重复帧处理、答案隔离、MCQ 标注边界、3DSR 分组、HTTP 推理、错误状态、恢复指纹和旧 VSI 协议。真实本地数据的验证结果见 [validation.md](docs/validation.md)。

新增 toolkit 没有自动下载/启动 encoder checkpoint，也没有替 Full / Small 生成所有新 benchmark 的框或提交新论文成绩。完成全量实验还需要对应的预测几何和正在服务的 reasoning LLM。指点/affordance/技能规划的适用限制见 benchmark matrix，不应把 31 项入口存在解释成仅凭 3D 框可充分解决 31 项任务。

代码来源及引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
