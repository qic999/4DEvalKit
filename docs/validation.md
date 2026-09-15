# 验证记录

验证日期：2026-09-13。以下是工具实现检查和已有预测的离线重评分，不是 Full / Small 新增全量推理成绩。

## 自动测试

`python -m pytest -q tests`：**60 passed**。覆盖：

- 31 项 adapter 无 GPU 导入，上游源文件与声明的 SHA256/import 改写一致。
- legacy 9D、10D 完整 quaternion、单位/参考系、非法尺寸/非有限值/非刚体相机检查。
- raw 动态轨迹尺度和坐标变换、重叠帧去重、时间顺序、跨 clip ID 要求。
- 答案/GT mask 等字段隔离，2D/no-observations 消融，lazy HF 媒体导出。
- VLM4D 数值零和重复正确选项；DSI 类别及四变体 robust 分组。
- 原生 PointBench 非严格 micro F1、QSpatial 单位及严格 factor-2 边界、3DSR 分组准确率。
- 本地 HTTP mock 的实际请求/响应、成功样本复用、缺失/截断输出计分、配置变化拒绝恢复、日志尾部恢复。
- legacy VSI 的 `qa_index` 与原生 `id` 不一致时仍正确关联，并校验 question / scene / options。

另通过 Python 编译检查、后台 Bash 语法检查、24 个 core-suite job 的命令生成。

## 真实本地数据检查

| 数据 | 可用规模 | 实际检查 |
|---|---:|---|
| VSI / ScanNet val88 | 2,071 QA / 88 scenes | 全部问题与已有预测框成功匹配；全部旧预测成功重评分 |
| STI-Bench | 2,064 QA | 全部准备成功，保留 8 类任务 |
| VLM4D real_mc | 1,371 QA | 全部准备成功，含数值 0 / 重复正确候选边界 |
| VLM4D synthetic_mc | 445 QA | 全部准备成功 |
| DSI std | 1,769 QA | 全部准备成功 |
| DSI all | 7,076 QA | 四种增强标注全部准备成功 |
| MMSI-Bench | 1,000 QA | 准备 2 题，并实际导出其嵌入式图像 |
| QSpatial_plus | 101 QA | 准备 2 题 |
| 3DSRBench circular TSV | 11,686 rows | `limit=1` 扩展为首个完整 base_qid 组，共 2 rows |

还读取了一个实际 SpatialEncoder 旧 raw JSONL 记录，验证其 homogeneous 4×4 intrinsics 与缺少 slot_id 时的 per-detection track-map。该单记录检查采用人为零时间，仅检查 schema/变换，不能作为实际 tracking 评测。

其余 Table 4 benchmark 已通过导入检查并保留上游实现，尚未在本机逐项跑完真实数据。尤其 affordance / grounding / 生成轨迹任务，需要补足相应输入信息后再评估。

## 旧预测重评分

输入：项目已有 `scannet_val88_gt2d_bbox_qwen35_9b_temp0_seed0_qwen_only.json`，2,071 条，全部匹配。没有新 LLM 请求。

| 协议 | 0–100 分数 | 解释 |
|---|---:|---|
| `project_vsi_acc_v1` | 51.3664896185 | 与原项目 acc.py 的直接计算一致，误差小于 1e-12（0–1 尺度） |
| `physbrain_4b37ca2` | 49.1322999755 | 同一组预测按上游 MRA / macro 协议重算 |

这两个数字不应彼此当作模型性能变化，也不能直接与 Table 4 完整 VSI 分数比较。`qwen_only` 文件里的 prompt 包含 3D 框，不能视为 no-box baseline。

## 后台与恢复

人工 dynamic fixture 的 Full / Small 两个 job 均成功；第二次 suite 运行每个 job 复用 2 条预测。对应的 suite preflight、JSONL journaling 和 CSV 汇总通过。样例 100 分来自人工构造答案，不是模型指标。

本机运行产物（均被 Git 忽略）：

- `results/verification_final/verification.json`：真实数据准备与旧 VSI parity 检查。
- `results/verification_final/vsi_replay_project.json`、`vsi_replay_physbrain.json`：逐题重评分。
- `results/verification_final/scores.csv`：带协议、数据源和 ScanNet filter 的汇总。
- `results/media_check/`：MMSI 两题实际媒体导出。
- `results/suite_smoke/`：两个 fixture job、恢复记录及 suite index。
- `logs/final_local_verification.log`：最终后台验证日志。
- `configs/local_verification.json`：本机路径配置，可另选新 output-dir 重跑。

验证环境：Python 3.13.9；NumPy 2.3.5；SciPy 1.16.3；datasets 4.8.5；PyArrow 21.0.0；Pillow 12.0.0；SacreBLEU 2.6.0；pytest 8.4.2；opencv-python-headless 4.14.0.94。模型服务不依赖这套环境加载 PyTorch。

以上记录对应最初的数据/评分适配验证阶段。当时尚未运行新的 encoder 推理。

## RGB pipeline validation

The RGB entry point is documented in [RGB evaluation](rgb_evaluation.md).
Native Full and Small inference has been exercised on single-image Q-Spatial
samples and timestamped STI samples. Validation checks frame-stage coverage,
native checkpoint compatibility, metric scale, camera motion transforms and
question-to-geometry mapping. The complete 101-question QSpatial_plus geometry
export has passed coverage checks for both variants.

The dense-scene RoPE memory change was checked on a native 493-observation clip
with exactly matching outputs. The depth-scaling initialization fix preserves
the existing large-foreground branch exactly and exercises the previously
failing small-foreground branch. Automated tests cover absolute video times,
multiview order, camera-motion compensation and explicit text precision.

Dataset preparation, geometry generation and completed QA are separate states.
Use each run's `status.json` and per-variant QA results to establish which full
splits actually finished; adapter availability and smoke tests do not establish
completed benchmark scores.
