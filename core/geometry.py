"""Validated object geometry, isolated from benchmark answer annotations."""
from __future__ import annotations

import math
from copy import deepcopy

from .io import read_json

BOX_KEYS = {"instance_id", "track_id", "category", "label", "center", "size", "quaternion_xyzw",
            "bbox_3d", "confidence", "bbox_2d", "visibility", "timestamp", "first_seen_time",
            "attributes", "view_id"}
SCENE_KEYS = {"schema_version", "scene_id", "units", "coordinate_frame", "objects", "tracks", "views",
              "cameras", "layout", "provenance", "initial_heading", "initial_heading_source",
              "camera_pose_source", "room_layout", "bboxes"}
FRAME_KEYS = {"timestamp", "center", "size", "quaternion_xyzw", "confidence", "visibility", "bbox_2d", "view_id"}
CAMERA_KEYS = {"timestamp", "view_id", "camera_to_world", "intrinsics", "image_size"}
FORBIDDEN = {"answer", "ground_truth", "gt_answer", "thought", "rationale", "correct_answer", "answer_value",
             "answer_unit", "target_mask", "gt_bbox", "correct_option", "solution"}


def check_no_answers(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN:
                raise ValueError(f"Answer-only field {key!r} is not allowed in model geometry")
            check_no_answers(child)
    elif isinstance(value, list):
        for child in value:
            check_no_answers(child)


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} contains nonfinite numbers")
    return result


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]


def normalize_camera(camera):
    if "timestamp" in camera:
        camera["timestamp"] = vector([camera["timestamp"]], 1, "camera timestamp")[0]
    if "camera_to_world" in camera:
        matrix = camera["camera_to_world"]
        if not isinstance(matrix, (list, tuple)) or len(matrix) != 4:
            raise ValueError("camera_to_world must be 4x4")
        matrix = [vector(row, 4, "camera_to_world row") for row in matrix]
        if any(abs(x-y) > 1e-5 for x,y in zip(matrix[3], [0,0,0,1])):
            raise ValueError("camera_to_world last row must be [0,0,0,1]")
        r = [row[:3] for row in matrix[:3]]
        det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1]) -
               r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0]) + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
        if abs(det-1) > 1e-3 or any(abs(sum(r[k][i]*r[k][j] for k in range(3)) - int(i == j)) > 1e-3
                                  for i in range(3) for j in range(3)):
            raise ValueError("camera_to_world rotation must be rigid (orthonormal, determinant +1)")
        camera["camera_to_world"] = matrix
    if "intrinsics" in camera:
        if len(camera["intrinsics"]) != 3:
            raise ValueError("intrinsics must be 3x3")
        camera["intrinsics"] = [vector(row, 3, "intrinsics row") for row in camera["intrinsics"]]
    if "image_size" in camera:
        camera["image_size"] = vector(camera["image_size"], 2, "image_size width,height")
        if min(camera["image_size"]) <= 0:
            raise ValueError("Image dimensions must be positive")


def normalize_box(box, index):
    if not isinstance(box, dict):
        raise ValueError("Each object must be a dictionary")
    unknown = set(box) - BOX_KEYS
    if unknown:
        raise ValueError(f"Unknown object fields: {sorted(unknown)}")
    obj = deepcopy(box)
    obj["instance_id"] = str(obj.pop("track_id", obj.get("instance_id", f"object_{index}")))
    obj["category"] = str(obj.pop("label", obj.get("category", "unknown")))
    if "bbox_3d" in obj:
        bbox = obj.pop("bbox_3d")
        if len(bbox) != 9:
            raise ValueError("Legacy bbox_3d must be 9D xyz,size,roll,pitch,yaw. Convert 10D boxes with explicit quaternion order first.")
        bbox = vector(bbox, 9, "bbox_3d")
        obj.update(center=bbox[:3], size=bbox[3:6], quaternion_xyzw=euler_to_quaternion(*bbox[6:]))
    obj["center"] = vector(obj.get("center"), 3, "center")
    obj["size"] = vector(obj.get("size"), 3, "size")
    if any(x <= 0 for x in obj["size"]):
        raise ValueError("Object dimensions must be positive")
    if "quaternion_xyzw" in obj:
        quat = vector(obj["quaternion_xyzw"], 4, "quaternion_xyzw")
        norm = math.sqrt(sum(x*x for x in quat))
        if norm < 1e-12:
            raise ValueError("Quaternion cannot be zero")
        obj["quaternion_xyzw"] = [x / norm for x in quat]
    for name in ["timestamp", "first_seen_time", "confidence"]:
        if name in obj:
            obj[name] = vector([obj[name]], 1, name)[0]
    if "bbox_2d" in obj:
        obj["bbox_2d"] = vector(obj["bbox_2d"], 4, "bbox_2d")
    return obj


def normalize_scene(scene, *, units=None, coordinate_frame=None):
    if isinstance(scene, list):
        scene = {"objects": scene}
    if not isinstance(scene, dict):
        raise ValueError("Scene must be an object or a legacy box list")
    if scene.get("schema_version", "4deval.geometry.v1") != "4deval.geometry.v1":
        raise ValueError("Unsupported geometry schema version")
    check_no_answers(scene)
    unknown = set(scene) - SCENE_KEYS
    if unknown:
        raise ValueError(f"Unknown scene fields: {sorted(unknown)}")
    scene = deepcopy(scene)
    scene["schema_version"] = "4deval.geometry.v1"
    scene["units"] = scene.get("units") or units
    scene["coordinate_frame"] = scene.get("coordinate_frame") or coordinate_frame
    if scene["units"] not in {"m", "cm", "mm"}:
        raise ValueError("Declare geometry units (m, cm, mm), including --units for legacy box files")
    if not isinstance(scene["coordinate_frame"], str) or not scene["coordinate_frame"].strip():
        raise ValueError("Declare coordinate_frame, e.g. world_z_up or camera_x_right_y_down_z_forward")
    scene["objects"] = [normalize_box(obj, i) for i, obj in enumerate(scene.pop("bboxes", scene.get("objects", [])))]
    ids = [obj["instance_id"] for obj in scene["objects"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate static instance IDs")
    track_ids = set()
    for track in scene.get("tracks", []):
        if set(track) - {"track_id", "category", "observations", "attributes"}:
            raise ValueError("Unknown track fields")
        track["track_id"] = str(track["track_id"])
        if track["track_id"] in track_ids:
            raise ValueError("Duplicate track IDs")
        track_ids.add(track["track_id"])
        if not track.get("observations"):
            raise ValueError("Tracks need observations")
        times = []
        for i, observation in enumerate(track["observations"]):
            if set(observation) - FRAME_KEYS:
                raise ValueError("Unknown track observation fields")
            normalized = normalize_box({**observation, "category": track.get("category", "unknown")}, i)
            normalized.pop("instance_id"); normalized.pop("category")
            if "timestamp" not in normalized:
                raise ValueError("Every track observation needs timestamp in seconds")
            times.append(normalized["timestamp"])
            track["observations"][i] = normalized
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Track timestamps must be strictly increasing; do not average motion into a static box")
    for camera in scene.get("cameras", []):
        if set(camera) - CAMERA_KEYS:
            raise ValueError("Unknown camera fields")
        normalize_camera(camera)
    for view in scene.get("views", []):
        if set(view) - {"view_id", "objects", "camera_to_world", "intrinsics", "image_size"}:
            raise ValueError("Unknown view fields")
        if "view_id" not in view:
            raise ValueError("Each view needs view_id")
        normalize_camera(view)
        view["objects"] = [normalize_box(obj, i) for i,obj in enumerate(view.get("objects", []))]
    return scene


class GeometryStore:
    def __init__(self, path, *, units=None, coordinate_frame=None):
        source = read_json(path)
        if isinstance(source, dict) and "scenes" in source:
            if source.get("schema_version", "4deval.geometry.v1") != "4deval.geometry.v1":
                raise ValueError("Unsupported geometry schema version")
            source = source["scenes"]
        if not isinstance(source, dict):
            raise ValueError("Geometry file must map sample/scene IDs to scene objects")
        self.scenes = {str(key): normalize_scene(value, units=units, coordinate_frame=coordinate_frame)
                       for key, value in source.items()}

    def lookup(self, sample_id, aliases=()):
        for key in [sample_id, *aliases]:
            if key is not None and str(key) in self.scenes:
                return str(key), self.scenes[str(key)]
        raise KeyError(f"No geometry for {sample_id}. Export the sample manifest to obtain stable IDs.")
