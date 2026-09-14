"""Convert SpatialEncoder merged boxes or raw per-frame outputs without losing rotation/motion.

Run as python -m scripts.convert_geometry --help from the repository root.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from core.geometry import normalize_scene, vector
from core.io import read_json, write_json, digest


def records(path):
    value = read_json(path)
    if isinstance(value, dict) and "bbox_3d" in value:
        value = [value]  # A one-line JSONL file is also valid JSON.
    if not isinstance(value, list):
        raise ValueError(f"Expected object records in {path}")
    return value


def merged_scene(rows, *, units, coordinate_frame, quaternion_order, provenance):
    objects = []
    for index, row in enumerate(rows):
        bbox = row["bbox_3d"]
        label = str(row.get("label", "unknown"))
        obj = {"instance_id": str(row.get("instance_id", f"{label}@{index}")),
               "category": str(row.get("category", label))}
        if len(bbox) == 10:
            if quaternion_order not in {"xyzw", "wxyz"}:
                raise ValueError("10D merged boxes require --quaternion-order xyzw|wxyz")
            bbox = vector(bbox, 10, "bbox_3d")
            quat = bbox[6:]
            obj.update(center=bbox[:3], size=bbox[3:6],
                       quaternion_xyzw=quat if quaternion_order == "xyzw" else quat[1:] + quat[:1])
        else:
            obj["bbox_3d"] = bbox
        for key in ["confidence", "bbox_2d", "first_seen_time"]:
            if key in row:
                obj[key] = row[key]
        objects.append(obj)
    return normalize_scene({"units": units, "coordinate_frame": coordinate_frame,
                            "objects": objects, "provenance": provenance})


def lookup_frame(mapping, row, path):
    candidates = [row.get("frame_path"), path.stem]
    for key in candidates:
        if key is not None and str(key) in mapping:
            return mapping[str(key)]
    raise ValueError(f"No explicit frame mapping for {row.get('frame_path', path.stem)}")


def raw_scene(input_dir, *, timestamps, poses, record_gt_poses, ids_global,
              track_map, metric_scale, coordinate_frame, provenance):
    import numpy as np
    from scipy.spatial.transform import Rotation
    if not math.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError("--metric-scale must be finite and positive")
    if not ids_global and track_map is None:
        raise ValueError("Raw boxes require --ids-global or --track-map; clip-local slots are not tracks")
    if (poses is None) == (not record_gt_poses):
        raise ValueError("Choose exactly one common-frame --poses map or --record-ground-truth-poses")
    root = Path(input_dir)
    files = sorted((root / "json").glob("*/*.jsonl"))
    if not files:
        raise ValueError(f"No json/<clip>/*.jsonl under {root}")
    tracks, cameras = {}, {}
    kept, duplicates = 0, 0
    for path in files:
        clip = path.parent.name
        for record_index, row in enumerate(records(path)):
            timestamp = float(lookup_frame(timestamps, row, path))
            if not math.isfinite(timestamp):
                raise ValueError("Timestamps must be finite seconds")
            slot = row.get("slot_id")
            if ids_global:
                if slot is None:
                    raise ValueError("--ids-global requires slot_id; use a per-detection --track-map for legacy raw records")
                track_id = str(slot)
            else:
                keys = [f"{clip}:{path.stem}:{record_index}"]
                if slot is not None:
                    keys.append(f"{clip}:{slot}")
                matches = [track_map[key] for key in keys if key in track_map]
                if not matches:
                    raise ValueError(f"Missing track identity mapping; expected one of {keys}")
                track_id = str(matches[0])
            pose = np.asarray(row["camera_pose"] if record_gt_poses else lookup_frame(poses, row, path), dtype=float)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
                raise ValueError("Camera poses must be rigid, metric camera-to-world transforms")
            bbox = vector(row["bbox_3d"], 6, "raw bbox_3d")
            qw, qx, qy, qz = vector(row["quaternion"], 4, "raw wxyz quaternion")
            rotation = Rotation.from_matrix(pose[:3, :3]) * Rotation.from_quat([qx, qy, qz, qw])
            conf = float(row.get("conf3d", 0))
            if not math.isfinite(conf) or conf < 0:
                raise ValueError("Raw conf3d must be finite and nonnegative")
            observation = {"timestamp": timestamp,
                "center": (pose[:3, :3] @ (np.asarray(bbox[:3]) * metric_scale) + pose[:3, 3]).tolist(),
                "size": (np.asarray(bbox[3:6]) * metric_scale).tolist(),
                "quaternion_xyzw": rotation.as_quat().tolist(), "confidence": conf / (1 + conf)}
            if "bbox_2d" in row:
                observation["bbox_2d"] = row["bbox_2d"]
            label = str(row.get("label", "unknown"))
            track = tracks.setdefault(track_id, {"track_id": track_id, "category": label, "observations": {}})
            if track["category"] != label:
                raise ValueError(f"Track {track_id} changes category; verify cross-clip identity mapping")
            old = track["observations"].get(timestamp)
            if old is None or observation["confidence"] > old["confidence"]:
                track["observations"][timestamp] = observation
            duplicates += int(old is not None)
            kept += int(old is None)
            camera = {"timestamp": timestamp, "camera_to_world": pose.tolist()}
            if "intrinsics" in row:
                intrinsic = np.asarray(row["intrinsics"], dtype=float)
                if intrinsic.shape == (4, 4):
                    if not np.allclose(intrinsic[:3, 3], 0) or not np.allclose(intrinsic[3], [0,0,0,1]):
                        raise ValueError("Unsupported nonhomogeneous 4x4 intrinsic matrix")
                    intrinsic = intrinsic[:3, :3]
                camera["intrinsics"] = intrinsic.tolist()
            if timestamp in cameras and not np.allclose(cameras[timestamp]["camera_to_world"], pose, atol=1e-5):
                raise ValueError("Overlapping clips have inconsistent world poses; align them before conversion")
            cameras[timestamp] = camera
    if not tracks:
        raise ValueError("Raw source contains no object observations")
    for track in tracks.values():
        track["observations"] = [value for _, value in sorted(track["observations"].items())]
    provenance = {**provenance, "raw_metric_scale": metric_scale,
                  "camera_pose_source": "ground_truth" if record_gt_poses else "external_common_frame",
                  "timestamp_map_digest": digest(timestamps), "pose_map_digest": digest(poses),
                  "identity_source": "declared_global_slot_id" if ids_global else "external_track_map",
                  "track_map_digest": digest(track_map), "observations": kept,
                  "overlapping_observations_deduplicated": duplicates}
    return normalize_scene({"units": "m", "coordinate_frame": coordinate_frame,
        "tracks": list(tracks.values()), "cameras": [v for _,v in sorted(cameras.items())], "provenance": provenance})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", required=True, choices=["legacy", "merged", "raw-tracks"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-id", help="Required for a single merged file or raw scene directory")
    parser.add_argument("--merged-name", help="For merged directories: exact filename under each scene directory")
    parser.add_argument("--quaternion-order", choices=["xyzw", "wxyz"])
    parser.add_argument("--units", choices=["m", "cm", "mm"])
    parser.add_argument("--coordinate-frame", required=True)
    parser.add_argument("--geometry-source", required=True, choices=["predicted", "oracle"])
    parser.add_argument("--encoder", required=True, help="spatial_encoder_v2, spatial_encoder_v2_small, or oracle")
    parser.add_argument("--checkpoint", help="Exact checkpoint identifier (logged, not sent to LLM)")
    parser.add_argument("--proposal-source", required=True, help="e.g. predicted_2d, gt_2d, none")
    parser.add_argument("--label-source", required=True, help="e.g. detector, gt_category")
    parser.add_argument("--camera-pose-source", help="Merged/legacy boxes: record whether original poses were predicted or GT")
    parser.add_argument("--timestamps", help="Raw: JSON frame_path or file stem -> seconds")
    parser.add_argument("--metric-scale", type=float, help="Raw: multiply model xyz and dimensions by this factor to get meters")
    pose = parser.add_mutually_exclusive_group()
    pose.add_argument("--poses", help="Raw: frame_path or file stem -> globally aligned metric camera-to-world 4x4")
    pose.add_argument("--record-ground-truth-poses", action="store_true")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--ids-global", action="store_true", help="Explicitly declare slot_id consistent across every input clip")
    identity.add_argument("--track-map", help="Raw: JSON 'clip:slot_id' or 'clip:frame_stem:record_index' -> global track ID")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output exists; use a new path")
    provenance = {"geometry_source": args.geometry_source, "encoder": args.encoder, "checkpoint": args.checkpoint,
                  "proposal_source": args.proposal_source, "label_source": args.label_source,
                  "camera_pose_source": args.camera_pose_source or "unspecified"}
    try:
        if args.format == "raw-tracks":
            if not args.scene_id or not args.timestamps or args.metric_scale is None:
                raise ValueError("raw-tracks requires --scene-id, --timestamps and --metric-scale")
            if args.units not in {None, "m"}:
                raise ValueError("Raw transforms produce meters; --units must be m or omitted")
            scenes = {args.scene_id: raw_scene(args.input, timestamps=read_json(args.timestamps),
                poses=read_json(args.poses) if args.poses else None, record_gt_poses=args.record_ground_truth_poses,
                ids_global=args.ids_global, track_map=read_json(args.track_map) if args.track_map else None,
                metric_scale=args.metric_scale, coordinate_frame=args.coordinate_frame, provenance=provenance)}
        elif args.format == "legacy":
            source = read_json(args.input)
            if not isinstance(source, dict):
                raise ValueError("Legacy input must map scene IDs to boxes")
            source = source.get("scenes", source)
            scenes = {}
            for key, value in source.items():
                scene = normalize_scene(value, units=args.units, coordinate_frame=args.coordinate_frame)
                scene["provenance"] = {**scene.get("provenance", {}), **provenance}
                scenes[str(key)] = scene
        else:
            path = Path(args.input)
            if path.is_file():
                if not args.scene_id:
                    raise ValueError("A single merged file requires --scene-id")
                files = [(args.scene_id, path)]
            else:
                if not args.merged_name:
                    raise ValueError("A merged directory requires --merged-name")
                files = [(p.parent.name, p) for p in sorted(path.glob(f"*/{args.merged_name}"))]
            if not files:
                raise ValueError("No merged scene files found")
            scenes = {key: merged_scene(records(path), units=args.units, coordinate_frame=args.coordinate_frame,
                      quaternion_order=args.quaternion_order, provenance=provenance) for key, path in files}
        write_json(args.output, {"schema_version": "4deval.geometry.v1", "scenes": scenes})
        print(f"Wrote {len(scenes)} scenes to {args.output}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"Geometry conversion: {exc}\n")


if __name__ == "__main__":
    main()
