"""Utilities for mapping multi-camera image paths to stitched-image identifiers.

ID format used throughout the DriveLM splits:
    "{scene_token}_{frame_token}_{qa_idx}"

    scene_token  — 32-char hex, identifies the nuScenes scene
    frame_token  — 32-char hex, identifies the keyframe within the scene
    qa_idx       — integer, QA pair index within that frame

To derive the stitched image filename from any record id:
    stem = id.rsplit("_", 1)[0]          # drop qa_idx
    stitched_image = stem + ".jpg"       # e.g. "<scene>_<frame>.jpg"
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


def build_cam_to_stitched_lookup(
    cam_jsonl,       # type: Union[str, Path]
    stitched_jsonl,  # type: Union[str, Path]
):
    # type: (...) -> Dict[str, str]
    """Build a lookup from multi-camera image paths to stitched image stem.

    Pairs entries positionally between *cam_jsonl* (6-path ``"image"`` list,
    integer ``"id"``) and *stitched_jsonl* (single ``"image"`` path, string
    ``"id"``).  Both files must be row-aligned (same order, same length).

    Returns:
        Dict mapping ``"|".join(sorted(image_paths))`` to the stitched image
        stem (filename without ``.jpg``), i.e. ``"{scene_token}_{frame_token}"``.
    """
    lookup = {}  # type: Dict[str, str]
    with open(cam_jsonl) as fc, open(stitched_jsonl) as fs:
        for line_cam, line_stitched in zip(fc, fs):
            cam_entry = json.loads(line_cam)
            stitched_entry = json.loads(line_stitched)
            key = "|".join(sorted(cam_entry["image"]))
            stem = Path(stitched_entry["image"]).stem
            lookup[key] = stem
    return lookup


def cam_paths_to_stitched_id(
    image_paths,  # type: List[str]
    lookup,       # type: Dict[str, str]
):
    # type: (...) -> Optional[str]
    """Return the stitched image stem for a list of per-camera image paths.

    Args:
        image_paths: The ``"image"`` list from a multi-cam JSONL entry
                     (order does not matter).
        lookup: Lookup built by :func:`build_cam_to_stitched_lookup`.

    Returns:
        Stitched image stem ``"{scene_token}_{frame_token}"`` or ``None`` if
        the frame is not in the lookup.
    """
    key = "|".join(sorted(image_paths))
    return lookup.get(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rewrite test.jsonl integer ids with proper {scene_token}_{frame_token}_{qa_idx} ids."
    )
    parser.add_argument("--orig-set", required=True,
                        help="row-aligned stitched JSONL (source of string ids), e.g. test_stitched.jsonl")
    parser.add_argument("--test", required=True,
                        help="test.jsonl with 6-cam image paths and integer ids — will be overwritten")
    args = parser.parse_args()

    lookup = build_cam_to_stitched_lookup(args.test, args.orig_set)
    print("Lookup built: %d unique frames" % len(lookup))

    fixed = []
    with open(args.test) as fc, open(args.orig_set) as fs:
        for line_cam, line_stitched in zip(fc, fs):
            entry = json.loads(line_cam)
            entry["id"] = json.loads(line_stitched)["id"]
            fixed.append(entry)

    Path(args.test).write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in fixed) + "\n")

    print("Done — %d entries written to %s" % (len(fixed), args.test))

# source vqa-ad-ctu-env/bin/activate
# python src/utils/helpers.py --orig-set <path/to/test_stitched.jsonl> --test data/drivelm_custom_split/test.jsonl