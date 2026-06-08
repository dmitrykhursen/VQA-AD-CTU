"""
Aggregate per-image annotation JSONs (from export_annotations.py) into
per-scene track files.

Input layout expected (produced by export_annotations.py):
  <input_dir>/<sensor_name>/<log_id>/<file_stem>.json

Output layout:
  scene mode  → <output_dir>/<scene_name>/tracks.json
  camera mode → <output_dir>/<camera_name>/<scene_name>/tracks.json
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

from nuscenes.nuscenes import NuScenes


def _build_pose_lookup(nusc: NuScenes) -> dict:
    """Map image basename → ego_pose_token for every sample_data record."""
    return {
        os.path.basename(sd["filename"]): sd["ego_pose_token"]
        for sd in nusc.sample_data
    }


def _append_frames(json_files, tracks: dict, nusc: NuScenes, pose_lookup: dict, camera_name=None):
    """Parse a sorted list of per-image JSONs and append track points."""
    img_path_base = ""

    for frame_idx, file_path in enumerate(json_files):
        with open(file_path) as f:
            data = json.load(f)

        if not img_path_base:
            img_path_base = data.get("img_path", "")

        for frame_data in data.get("results", []):
            image_name = frame_data.get("image_name", "unknown_image")

            ego_translation = [None, None, None]
            ego_rotation = [None, None, None, None]
            if image_name in pose_lookup:
                pose = nusc.get("ego_pose", pose_lookup[image_name])
                ego_translation = pose["translation"]
                ego_rotation = pose["rotation"]
            else:
                print(f"Warning: no ego pose found for {image_name}")

            feat_2d = {f["object_id"]: f for f in frame_data.get("2d_feat", [])}
            feat_3d = {f["object_id"]: f for f in frame_data.get("3d_feat", [])}

            for obj_id in set(feat_2d) | set(feat_3d):
                if obj_id not in tracks:
                    raw_cat = feat_3d.get(obj_id, feat_2d.get(obj_id, {})).get("category_name", "unknown")
                    category = raw_cat.split(".")[-1] if "." in raw_cat else raw_cat
                    tracks[obj_id] = {"object_id": obj_id, "category": category, "track": []}

                f2d = feat_2d.get(obj_id, {})
                f3d = feat_3d.get(obj_id, {})

                c3d = f3d.get("bbox_center_3d", [None, None, None])
                c2d = f2d.get("bbox_center", [None, None])
                depth = f3d.get("depth")

                point = {
                    "frame": frame_idx,
                    "frame_name": image_name,
                    "x": round(c3d[0], 3) if c3d[0] is not None else None,
                    "y": round(c3d[1], 3) if c3d[1] is not None else None,
                    "z": round(c3d[2], 3) if c3d[2] is not None else None,
                    "center_2d_px": [
                        int(c2d[0]) if c2d[0] is not None else None,
                        int(c2d[1]) if c2d[1] is not None else None,
                    ],
                    "depth": round(depth, 2) if depth is not None else None,
                    "ego_translation": [round(t, 3) if t is not None else None for t in ego_translation],
                    "ego_rotation": [round(r, 3) if r is not None else None for r in ego_rotation],
                }
                if camera_name:
                    point["camera"] = camera_name

                tracks[obj_id]["track"].append(point)

    return img_path_base


def _save_tracks(tracks: dict, output_folder: str, img_path_base: str) -> None:
    if not tracks:
        return
    os.makedirs(output_folder, exist_ok=True)

    for obj in tracks.values():
        obj["track"].sort(key=lambda p: p["frame"])

    out = {
        "date_time_tracks_generated": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "img_path_base": img_path_base,
        "tracks": list(tracks.values()),
    }

    out_path = os.path.join(output_folder, "tracks.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {len(tracks)} objects → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-image annotation JSONs into per-scene track files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, help="Root dir of per-image annotation JSONs.")
    parser.add_argument("--output_dir", required=True, help="Root dir for output track JSONs.")
    parser.add_argument(
        "--organize_by",
        choices=["scene", "camera"],
        default="scene",
        help="Group output by scene (one tracks.json per scene) or by camera.",
    )
    parser.add_argument("--nusc_dataroot", required=True, help="NuScenes dataset root.")
    parser.add_argument("--nusc_version", default="v1.0-trainval", help="NuScenes version string.")
    args = parser.parse_args()

    print(f"Loading nuScenes ({args.nusc_version}) from {args.nusc_dataroot} ...")
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_dataroot, verbose=False)

    print("Building image → ego-pose lookup ...")
    pose_lookup = _build_pose_lookup(nusc)

    # Collect files grouped by (scene, camera) or (camera, scene)
    groupings: dict = defaultdict(lambda: defaultdict(list))

    for cam_dir in sorted(os.listdir(args.input_dir)):
        cam_path = os.path.join(args.input_dir, cam_dir)
        if not os.path.isdir(cam_path):
            continue
        for scene_dir in sorted(os.listdir(cam_path)):
            scene_path = os.path.join(cam_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            files = sorted(
                os.path.join(scene_path, f)
                for f in os.listdir(scene_path)
                if f.endswith(".json") and f != "merged_processed.json"
            )
            if not files:
                continue
            if args.organize_by == "scene":
                groupings[scene_dir][cam_dir] = files
            else:
                groupings[cam_dir][scene_dir] = files

    if args.organize_by == "scene":
        print(f"Scene mode → {args.output_dir}/<scene>/tracks.json")
        for scene_name, cameras in groupings.items():
            tracks: dict = {}
            img_base = ""
            for cam_name, json_files in cameras.items():
                img_base = _append_frames(json_files, tracks, nusc, pose_lookup, camera_name=cam_name) or img_base
            _save_tracks(tracks, os.path.join(args.output_dir, scene_name), img_base)
    else:
        print(f"Camera mode → {args.output_dir}/<camera>/<scene>/tracks.json")
        for cam_name, scenes in groupings.items():
            for scene_name, json_files in scenes.items():
                tracks = {}
                img_base = _append_frames(json_files, tracks, nusc, pose_lookup)
                _save_tracks(tracks, os.path.join(args.output_dir, cam_name, scene_name), img_base)

    print("Done.")


if __name__ == "__main__":
    main()
