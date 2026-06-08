"""
Export per-image 2D + 3D annotations from a NuScenes split.

Outputs one JSON per camera image under:
  <output_dir>/<sensor_name>/<log_id>/<file_stem>.json

Each JSON contains:
  image_name, time_stamp,
  2d_feat  – list of {bbox, bbox_center, object_id, score, category_name}
  3d_feat  – list of {bbox_3d, bbox_center_3d, object_id, category_name,
                       n_lidar_points, depth, mean_3d_point, median_3d_point}
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box, view_points

CAMERA_SENSORS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Per-worker NuScenes instance (initialised once per worker process)
_worker_nusc: Optional[NuScenes] = None
_worker_instance_id_map: dict = {}


def _worker_init(dataroot: str, version: str, instance_id_map: dict) -> None:
    global _worker_nusc, _worker_instance_id_map
    _worker_nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)
    _worker_instance_id_map = instance_id_map


def _project_box_to_2d(
    global_box,
    pose_rec: dict,
    cs_rec: dict,
    intrinsic: np.ndarray,
    imsize: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    view_box = global_box.copy()
    view_box.translate(-np.array(pose_rec["translation"]))
    view_box.rotate(Quaternion(pose_rec["rotation"]).inverse)
    view_box.translate(-np.array(cs_rec["translation"]))
    view_box.rotate(Quaternion(cs_rec["rotation"]).inverse)

    corners_3d = view_box.corners()
    in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
    corners_3d_front = corners_3d[:, in_front]

    corner_coords = view_points(corners_3d_front, intrinsic, normalize=True).T[:, :2].tolist()

    polygon = MultiPoint(corner_coords).convex_hull
    canvas = box(0, 0, imsize[0], imsize[1])

    if not polygon.intersects(canvas):
        return None

    intersection = polygon.intersection(canvas)
    if intersection.is_empty:
        return None

    if intersection.geom_type == "Polygon":
        coords = np.array(intersection.exterior.coords)
    elif intersection.geom_type == "MultiPolygon":
        coords = np.concatenate([np.array(p.exterior.coords) for p in intersection.geoms])
    else:
        return None

    if len(coords) == 0:
        return None

    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    return float(min_x), float(min_y), float(max_x), float(max_y)


def _process_sample(nusc: NuScenes, sample: dict, visibilities: list, instance_id_map: dict) -> List[dict]:
    """Return a list of per-image result dicts for one NuScenes sample."""
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd_lidar = nusc.get("sample_data", lidar_token)
    cs_lidar = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])
    pose_lidar = nusc.get("ego_pose", sd_lidar["ego_pose_token"])

    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_lidar["filename"]))
    pc.rotate(Quaternion(cs_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_lidar["translation"]))
    pc.rotate(Quaternion(pose_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(pose_lidar["translation"]))
    points_global = pc.points[:3, :]

    results = []

    for sensor_name in CAMERA_SENSORS:
        if sensor_name not in sample["data"]:
            continue

        cam_token = sample["data"][sensor_name]
        sd_cam = nusc.get("sample_data", cam_token)
        cs_cam = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
        pose_cam = nusc.get("ego_pose", sd_cam["ego_pose_token"])
        intrinsic = np.array(cs_cam["camera_intrinsic"])
        imsize = (sd_cam["width"], sd_cam["height"])

        s_rec = nusc.get("sample", sd_cam["sample_token"])
        ann_recs = [nusc.get("sample_annotation", t) for t in s_rec["anns"]]
        ann_recs = [a for a in ann_recs if str(a["visibility_token"]) in visibilities]

        cam_global_pos = np.array(pose_cam["translation"]) + Quaternion(
            pose_cam["rotation"]
        ).rotate(np.array(cs_cam["translation"]))

        feat_2d, feat_3d = [], []

        for ann in ann_recs:
            global_box = nusc.get_box(ann["token"])

            coords_2d = _project_box_to_2d(global_box, pose_cam, cs_cam, intrinsic, imsize)
            if coords_2d is None:
                continue

            min_x, min_y, max_x, max_y = coords_2d
            w, h = max_x - min_x, max_y - min_y
            cx, cy = min_x + w / 2.0, min_y + h / 2.0
            obj_id = instance_id_map.get(ann["instance_token"], ann["instance_token"])

            mask = points_in_box(global_box, points_global)
            box_pts = points_global[:, mask]
            n_pts = box_pts.shape[1]

            depth = float(np.linalg.norm(global_box.center - cam_global_pos))
            mean_pt = np.mean(box_pts, axis=1).tolist() if n_pts > 0 else None
            median_pt = np.median(box_pts, axis=1).tolist() if n_pts > 0 else None
            yaw = float(Quaternion(global_box.orientation).yaw_pitch_roll[0])

            feat_2d.append({
                "bbox": [cx, cy, w, h],
                "bbox_center": [cx, cy],
                "object_id": obj_id,
                "score": 1.0,
                "category_name": ann["category_name"],
            })
            feat_3d.append({
                "bbox_3d": [*global_box.center.tolist(), *global_box.wlh.tolist(), yaw],
                "bbox_center_3d": global_box.center.tolist(),
                "object_id": obj_id,
                "category_name": ann["category_name"],
                "n_lidar_points": int(n_pts),
                "depth": depth,
                "mean_3d_point": mean_pt,
                "median_3d_point": median_pt,
            })

        filename_rel = sd_cam["filename"]
        image_basename = os.path.basename(filename_rel)
        file_stem = os.path.splitext(image_basename)[0]

        parts = image_basename.split("__")
        log_id = parts[0].split("+")[0] if len(parts) >= 2 else "UNKNOWN_LOG"
        sensor_dir = parts[1] if len(parts) >= 2 else "UNKNOWN_SENSOR"

        results.append({
            "_meta": {
                "sensor_dir": sensor_dir,
                "log_id": log_id,
                "file_stem": file_stem,
                "img_path": os.path.dirname(os.path.join(nusc.dataroot, filename_rel)),
            },
            "image_name": image_basename,
            "time_stamp": str(sd_cam["timestamp"]),
            "2d_feat": feat_2d,
            "3d_feat": feat_3d,
        })

    return results


def _worker_process_sample(sample_token: str, output_dir: str, visibilities: list, gen_ts: str, gen_dt: str) -> int:
    """Top-level function executed in worker processes."""
    sample = _worker_nusc.get("sample", sample_token)
    image_results = _process_sample(_worker_nusc, sample, visibilities, _worker_instance_id_map)

    for res in image_results:
        meta = res.pop("_meta")
        target_dir = os.path.join(output_dir, meta["sensor_dir"], meta["log_id"])
        os.makedirs(target_dir, exist_ok=True)

        output = {
            "name": "NuScenes_GT_Custom",
            "time_stamp_generation": gen_ts,
            "date_time": gen_dt,
            "img_path": meta["img_path"],
            "model": "NuScenes_GT_Custom",
            "model_version": "v1.0",
            "output_type": "2D feat, 3D feat",
            "results": [res],
        }

        out_path = os.path.join(target_dir, f"{meta['file_stem']}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    return len(image_results)


def main(args: argparse.Namespace) -> None:
    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    n_samples = len(nusc.sample)
    print(f"Processing {args.version} | {n_samples} samples | {args.num_workers} worker(s)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build deterministic instance_token → int ID map (stable across workers)
    instance_id_map = {inst["token"]: i + 1 for i, inst in enumerate(nusc.instance)}

    sample_tokens = [s["token"] for s in nusc.sample]
    if args.image_limit != -1:
        sample_tokens = sample_tokens[: args.image_limit]

    gen_ts = str(int(time.time() * 1e6))
    gen_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.num_workers <= 1:
        # Single-process path: reuse the already-loaded nusc object
        _worker_init(args.dataroot, args.version, instance_id_map)
        for token in tqdm(sample_tokens, desc="samples"):
            _worker_process_sample(token, args.output_dir, args.visibilities, gen_ts, gen_dt)
    else:
        initargs = (args.dataroot, args.version, instance_id_map)
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=initargs,
        ) as pool:
            futures = {
                pool.submit(
                    _worker_process_sample,
                    token,
                    args.output_dir,
                    args.visibilities,
                    gen_ts,
                    gen_dt,
                ): token
                for token in sample_tokens
            }
            with tqdm(total=len(futures), desc="samples") as pbar:
                for fut in as_completed(futures):
                    fut.result()  # re-raises any worker exception
                    pbar.update(1)

    print(f"Done. {len(sample_tokens)} samples → {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export 2D/3D annotations with LiDAR features from NuScenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataroot", required=True, help="NuScenes dataset root directory.")
    parser.add_argument("--version", default="v1.0-trainval", help="Dataset version.")
    parser.add_argument("--output_dir", required=True, help="Root directory for JSON outputs.")
    parser.add_argument(
        "--visibilities",
        nargs="+",
        default=["", "1", "2", "3", "4"],
        help="Visibility tokens to include ('' = unknown, 1–4 = least to fully visible).",
    )
    parser.add_argument(
        "--image_limit",
        type=int,
        default=-1,
        help="Max samples to process (-1 = all).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Worker processes for parallel sample processing.",
    )
    main(parser.parse_args())
