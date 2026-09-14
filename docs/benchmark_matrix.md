# Benchmark 覆盖及使用边界

依据 PhysBrain 1.5 技术报告 Table 4、§5.1、Appendix B.1，以及 [官方代码](https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit/tree/4b37ca2f184bd76b98827481a6739f89ea06d730)。这里的优先级是针对 **预测物体 3D 框 → LLM reasoning** 的判断，不是 benchmark 作者对 SpatialEncoder 兼容性的声明。

两种 SpatialEncoder 都可以输出本库需要的几何接口。可比实验需固定：输入帧、2D proposals、类别来源、LLM、prompt、采样参数、单位、坐标系和几何导出策略；变化项为 encoder/checkpoint。GT 框、GT 2D proposal、GT 类别、GT camera pose 应分项记录。

## Table 4：全部 28 项

| Benchmark | 场景 / 任务 | 建议 | 框输入还需要什么 |
|---|---|---|---|
| BLINK | 静态视觉空间；本表用计数/深度/关系 | 主评测 | 类别、实例、多图对应关系 |
| CV-Bench | 静态 2D / 3D 感知 | 主评测 | 相机参考系、实例计数、深度定义 |
| 3DSRBench | 静态单图 3D 关系与视角推理 | 主评测 | 完整旋转、语义朝向；圆周/镜像变体须分别生成框 |
| EmbSpatial-Bench | 静态具身空间关系 | 主评测 | 类别与参考视图 |
| MindCube | 多视图空间整合 | 主评测 | 每个 view 的几何、跨视图 ID、相机关系 |
| MMSI-Bench | 多图空间理解 | 主评测 | 视图关联及参考系；GT thought 严禁进入 reasoner |
| Q-Spatial-Bench | 单图定量距离 | 主评测 | 公制尺度、单位、对象对应；默认 QSpatial_plus |
| RoboSpatial-Home | 静态 context 指点 + compatibility/configuration 判断 | 扩展 | image-space points、相机投影及物理可放置性 |
| SAT | 动作条件/视角变化空间推理 | 主评测，单列 | 输入视图和问题中的动作条件；不等同真实运动视频 |
| VSI-Bench | 视频观察静态室内环境 | 主评测，static video | 场景对象、第一出现时间、相机路径；部分题还需要布局/初始朝向 |
| ViewSpatial-Bench | 多视图、视角转换 | 主评测 | 相机位姿、跨视图身份、对象朝向 |
| COSMOS | 视频具身常识与推理 | 扩展 | 动作、交互及外观语义；仅框不一定可回答 |
| EgoPlan-Bench2 | 第一视角视频任务规划 | 扩展 | 任务历史、状态变化和下一步动作语义 |
| ERQA | 具身推理 | 扩展 | 可见状态、对象属性及行为常识 |
| ERQA-PLUS | 具身推理扩展 | 扩展 | 使用 huggingdas/erqa-plus；不混同名称相似数据 |
| RoboVQA | 视频开放式问答 | 扩展 | 16 帧时序及动作语义，保留 BLEU 协议 |
| VLABench | 离线机器人技能规划 | 扩展 | 技能词表、参数、前提状态；非 simulator rollout |
| Part-Affordance | 部件指点 | 扩展 | 部件/功能面；整物体 3D 框通常不够 |
| PIOBench | 具身指点 S1 / S2 | 扩展 | 原图坐标、指令语义、可操作区域 |
| PixMo-Points | 对象定位 / 计数 | 扩展 | 2D 实例位置与可见性 |
| PointBench | 指点 / 计数 | 扩展 | 对象像素位置、数量；计分 mask 仅供 scorer |
| RefSpatial-Bench | 空间指代表达与放置点 | 扩展 | 指代对象、像素投影、有效区域 |
| RoboAfford | affordance grounding | 扩展 | 接触点/部件/动作可行性 |
| RoboRefit | 纠正后的机器人 grounding | 扩展 | corrected split、属性与操作部位 |
| VABench-Point | 指令条件指点 | 扩展 | 目标区域、视图像素坐标 |
| Where2Place | 放置点 | 扩展 | 支撑面、碰撞、空闲空间；单个 object center 不够 |
| ShareRobot-Traj | 单图 → 生成 2D 操作轨迹 | 扩展，单列 | 指令、image-space 轨迹；不是视频 tracking |
| VABench-V-Trace | 单图 → 生成 visual trace | 扩展，单列 | 同上，不能作为真实物体运动理解的直接证据 |

本库保留上述 28 项的原生 adapter 和 scorer，并提供统一文本入口。扩展任务是否有足够观测信息需要按具体题型确认；没有部件/外观/相机信息时，低分可能反映表示缺少信息，而非 LLM 缺少空间 reasoning 能力。

## 动态补充：不在 Table 4

| Benchmark | 优先级 | 用途 | 当前接口 |
|---|---|---|---|
| [STI-Bench](https://mint-sjtu.github.io/STI-Bench.io/) | 主评测 | 视频定量空间、位移/路径/速度、相机运动等 | qa.parquet，逐题 task/scene 分项；秒制 timestamps |
| [VLM4D](https://vlm4d.github.io/) | 主评测 | 真实及合成运动对象的时空推理 | real_mc / synthetic_mc 分开；确定性选项协议 |
| [DSI-Bench](https://dsibench.github.io/) | 动态扩展 | 动态空间与倒放/镜像鲁棒性 | std / reverse / hflip / reverse_hflip；四变体完整才有 robust score |

动态对象应使用 `tracks`，相机运动单独使用 `cameras`；不能用 camera-frame 位置差直接当 world-frame 物体位移，也不能把“相机在动、物体不动”的 VSI 当作完整动态评测。

建议主表按五组展示：静态单图、静态多视图、静态视频、动作条件空间推理、真实/合成动态。至少保留 Full / Small、predicted 3D / GT 3D oracle、2D boxes / no observations 对照。VLM 的 RGB-only 和 RGB+boxes baseline 可另行产生原生输出后重评分。Grounding 和生成轨迹用独立扩展表，避免一个 overall 掩盖输入信息差异。

执行用数据源和 split 以 [registry](../scripts/benchmark_registry.py) 为准；评分细节见 [metric_protocols.md](metric_protocols.md)。
