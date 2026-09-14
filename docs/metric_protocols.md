# 评分协议与可比性

Table 4 的 benchmark 数据/评分代码固定于 [PhysBrainEvalKit revision 4b37ca2](https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit/tree/4b37ca2f184bd76b98827481a6739f89ea06d730)。`benchmark/physbrain` 和 `core/physbrain` 仅修改 import 路径。新 runner 的输入改成预测几何文本，默认 temperature=0、seed=0、max_tokens=512；这不是对原 VLM 输入/采样配置的复刻。MMSI 上游 suite 有独立采样配置，这里不默认沿用。

| Benchmark | 本库采用的协议 |
|---|---|
| BLINK | Counting、Relative_Depth、Spatial_Relation 三个子集 |
| CV-Bench、EmbSpatial、MindCube、MMSI、SAT、ViewSpatial | 原生选择题 scorer；统计字段保留 |
| 3DSRBench | Circular + FlipEval；同一 base_qid 的所有变体都正确才算对。`--limit` 扩展到完整组，不取单变体冒充 grouped accuracy |
| MindCube | 默认 TinyBench，原生 `MindCube_tinybench.jsonl`；不可把另一个同名 tiny parquet 自动视为同协议 |
| Q-Spatial-Bench | 默认 QSpatial_plus，单位归一化后的 factor-2 success；保留 parse rate |
| VSI physbrain | 上游 MRA、相对方向 easy/medium/hard 均值合并，再对任务作 macro average |
| VSI project | 项目旧 acc.py 原样保存；10 个 `0.5+0.05*i` 阈值、严格 `<`；方向题先合并样本，overall 为 sample-weighted mean |
| COSMOS、EgoPlan-Bench2、ERQA、ERQA-PLUS | 原生答案解析及 accuracy；ERQA-PLUS 用 `huggingdas/erqa-plus` |
| RoboVQA | 16 帧版本、mean per-sample SacreBLEU，0–100 |
| VLABench | 离线 JSON skill sequence 分项分数，0–100，不是模拟器成功率 |
| 9 项 grounding + RoboSpatial | 固定 point metrics；非严格 micro F1。RoboSpatial 为 point + binary 样本数加权 overall |
| ShareRobot-Traj、VABench-V-Trace | 原生 normalized RMSE 字段实为误差；展示分数用 `100 - error`，保留 valid / total |
| STI-Bench | 原生 qa.parquet，全任务直接选项准确率；另列 task、scene |
| VLM4D | `vlm4d_direct_choice_v1`，real/synthetic 分开。不是原论文自由文本 + judge + manual 协议 |
| DSI-Bench | 单变体选项准确率；all 模式完整四变体后才报告至少 1/2/3/4 个变体正确的 robust accuracy |

VSI 上游代码使用 `np.linspace(.5,.95,int((.95-.5)/.05+2))` 和 `<=`，不等同于旧项目 10 阈值版本；这里保留代码行为。旧项目协议也保留它原有的答案文本解析。要复核原结果用 `--vsi-metric-protocol project`，不要以为只改 overall 的平均方式就能完全复现。

VLM4D real 的 `validation_633` 原始标注中 C、D 文本相同且均等于正确答案；本库保留题目并接受这两个选项，结果的 `ground_truth` 明确记录列表。数值答案 `0` 作为有效答案，不能用 truthiness 当作空值。

动态分项：STI 保留 8 种原生任务和 scene；DSI 列出固定/移动相机下对象运动、静态/动态场景下相机运动、对象–相机距离、对象–相机朝向。VLM4D 沿用作者统计脚本的 real exocentric / egocentric ID 分组，并在 synthetic 分开 directional / false-positive；这些 scoring strata 不进入模型 prompt 或媒体 manifest。

Point 坐标沿用 Qwen-style 0–1000 约定；几何输入中的 `bbox_2d` 则是像素 xyxy。若需要从 3D 推理到 image-space point，必须提供对应视图的 intrinsics / extrinsics / image_size，不能用 GT target mask 或 GT point 作为 LLM 输入。完整协议见 [final_point_metrics_protocol.md](final_point_metrics_protocol.md)。

完成状态与评分分开记录：缺失预测、API 错误、`finish_reason=length` 等会作为空输出交给 scorer。原生选择题/数值/指点任务通常由此得到 0 分。原生轨迹 scorer 只在有效轨迹上平均距离，因此本库额外展示 `valid_samples` 和 `total_samples`；不能仅报这个条件平均分而隐藏低输出成功率。本库不静默改写这一原生指标。

所有最终 `primary_metric.score_100` 越大越好；原始统计仍完整保留。`selection_is_partial` 只表示相对当前 `--data` / source filter 的选择是否部分；它不会证明本地镜像就是官方完整数据。当前 ScanNet 2071 QA 即使该字段为 false，仍只是 VSI 的一个来源子集。

不能把“视频”直接等同“物体运动”：VSI 主要是静态场景里的移动相机；SAT 包含动作条件空间推理；生成二维操作轨迹也不等于追踪真实视频运动。请按 `static`、`multiview`、`static_video`、`action_conditioned`、`dynamic` 分别报告。
