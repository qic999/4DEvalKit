# 几何输入契约

顶层用 `{"scenes": {"sample_or_scene_id": scene}}`，也兼容直接 ID → scene 的字典。schema 为 `4deval.geometry.v1`。单位必须明确为 `m` / `cm` / `mm`；同一个 scene 的对象中心、尺寸和 camera translation 使用同一单位。尺寸是沿 OBB 局部三个轴的**完整边长**，不是半长。`quaternion_xyzw` 是 `[x,y,z,w]`；9D legacy Euler 使用 roll/pitch/yaw，弧度，`Rz(yaw) Ry(pitch) Rx(roll)`。

```json
{
  "scenes": {
    "vsi_bench:0": {
      "schema_version": "4deval.geometry.v1",
      "units": "m",
      "coordinate_frame": "world_z_up",
      "objects": [
        {
          "instance_id": "chair_1",
          "category": "chair",
          "center": [1.2, 0.5, 0.6],
          "size": [0.5, 0.5, 1.1],
          "quaternion_xyzw": [0, 0, 0, 1],
          "confidence": 0.9,
          "first_seen_time": 1.25
        }
      ],
      "provenance": {
        "encoder": "spatial_encoder_v2_small",
        "checkpoint": "exact-checkpoint-identifier",
        "geometry_source": "predicted",
        "proposal_source": "gt_2d",
        "label_source": "gt_category",
        "camera_pose_source": "ground_truth"
      }
    }
  }
}
```

`instance_id` 必须区分同类物体，不能把所有 chair 合成一个实例。9D legacy 只有 `label` 时分配稳定列表 ID；不会猜测空间合并。新静态合并转换也不删除原 label 的实例后缀；若已知规范类别，可在源记录提供独立 `category`。

`provenance` 写入结果审计记录，不送入 LLM。必须如实记录来源，代码只能检查结构，不能证明用户提交的框确实来自预测。`answer`、`ground_truth`、`thought`、`rationale`、`target_mask` 等字段拒绝出现在几何输入中；GT 3D oracle 应提供相同几何字段并在 provenance 标记 oracle，而不是嵌入 benchmark 答案。

## 动态及多视图

动态 scene 可不含 `objects`，改用：

```json
{
  "units": "m",
  "coordinate_frame": "world_z_up",
  "tracks": [
    {
      "track_id": "cup_1",
      "category": "cup",
      "observations": [
        {"timestamp": 0.0, "center": [0,0,1], "size": [0.1,0.1,0.2], "quaternion_xyzw": [0,0,0,1]},
        {"timestamp": 0.5, "center": [0.2,0,1], "size": [0.1,0.1,0.2], "quaternion_xyzw": [0,0,0,1]}
      ]
    }
  ],
  "cameras": [
    {"timestamp": 0.0, "camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
    {"timestamp": 0.5, "camera_to_world": [[1,0,0,0.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
  ]
}
```

每个 track 的时间严格递增，单位秒，必须与原题的视频时间原点一致。若视频裁剪后时间清零，需恢复原时间或同时明确转换题目时间，不能只改其中一边。遮挡可以少观测，但不能伪造位置插值。旋转是几何 OBB 朝向，不自动等于人/车等对象的语义前向。

多视图 scene 可使用 `views` 列表，每项包含 `view_id`、`objects`，可附 `camera_to_world`、3×3 `intrinsics`、`image_size:[width,height]`。各对象的 `view_id` / track 观测的 `view_id` 应对应真实输入视图。相机关系未知时使用明确的 view-local frame 描述，不可把不同相机局部框放进同一 world 坐标系直接算距离。

可选 `bbox_2d` 统一为当前视图像素 `[xmin,ymin,xmax,ymax]`。2D 消融保留 ID、类别、可见性、时间和真实 2D 框，去掉 center/size/quaternion/相机/布局。`none` 消融完全去掉场景观测。不能在 attributes 等自由字段藏回本来应被消融的几何量。

## 从两个模型现有输出转换

- `--format legacy`：读现有 9D scene dictionary，要求单位/坐标系，增加来源信息。
- `--format merged`：读 `merge_json.py` 的逐行 `label,bbox_3d`。10D 是 xyz、三个完整尺寸、四元数；必须明确 quaternion order。保留 roll/pitch 信息，不做 yaw-only 简化。
- `--format raw-tracks`：读单场景 `json/<clip>/*.jsonl`，原始 `bbox_3d` 为 6D，`quaternion` 固定为当前 encoder 的 wxyz 格式。需要明确尺度、时间、ID、世界位姿。

如果旧 `convert.py` 已经把框压成 yaw-only 的 9D 表示，无法从该文件恢复丢失的 roll/pitch；应从原 10D 合并文件或 raw 输出重新导出。

raw 输入的辅助 JSON：

```json
{"path/to/frame_000.png": 0.0, "path/to/frame_015.png": 0.5}
```

这是 `--timestamps`，键匹配原记录的 `frame_path`，也可用输出 JSONL 的完整文件 stem。`--poses` 使用同样的键，值为全局对齐且 translation 已是米的 4×4 camera-to-world。变换为 `world_center = R @ (raw_center * metric_scale) + translation`，尺寸乘同一 scale，旋转为 `R_camera @ R_box`。没有额外轴交换，提供的相机位姿必须采用原始模型框所在的相机轴约定。

```json
{"0:slot_1": "global_chair_1", "1:slot_7": "global_chair_1"}
```

这是 `--track-map`，键为 `clip_index:slot_id`。模型自己的不同 clip 的 slot 可以复用；只有事先确认全场景 ID 一致才使用 `--ids-global`。若想接入自动关联，应在外部 tracker 输出这个映射。raw converter 按 `(track_id,timestamp)` 去重重叠帧并取最高 conf3d 的观测，保留不同 timestamp 的全部移动，不做时间平均；同一时刻相机位姿不一致时直接报错。

部分旧 raw JSONL 没有 `slot_id`。这种情况用更精确的映射键 `clip_index:frame_stem:record_index`，最后一项为文件内记录的零起始索引；不要把同类 label 当唯一 ID。当前模型导出的 homogeneous 4×4 intrinsics 会取有效左上 3×3 保存，其他非标准扩展矩阵报错。

## 导出 encoder 输入媒体

```bash
python eval_mmsi_bench.py --data /path/to/MMSI-Bench --limit 2 \
  --export-manifest data/manifests/mmsi_smoke.json \
  --export-media-dir data/media/mmsi_smoke
```

嵌入式图片/帧存 PNG，已编码 MP4/WebM 原样保存；已有路径/URL 保留引用，不自动下载。lazy HF images/video 从 Arrow 列提取，避免误把标注中的 thought/mask 当作视觉输入。导出文件名的顺序只表示顺序，不伪造 FPS 或 timestamp；时间来自原视频/数据元信息。未知媒体格式报错，需使用原数据 loader。

同一导出目录不覆盖已有文件。不同 benchmark 的预处理可能包含镜像、缩放、裁剪或多图顺序；encoder 应使用这里实际 prepared media，并沿用 manifest 的 sample ID。对按增强计分的数据，仅靠原图文件路径关联框是不够的。
