"""Run the Analyze node on a real image and save compact person diagnostics."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation/multi_person_result.json"))
    parser.add_argument("--disable-xformers", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    comfy_root = project.parents[1]
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(project.parent))
    if args.disable_xformers:
        sys.argv = [sys.argv[0], "--disable-xformers"]
        import comfy.options
        comfy.options.enable_args_parsing()
    nodes = importlib.import_module(f"{project.name}.nodes")

    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32)
    pose_image = torch.from_numpy(rgb / 255.0).unsqueeze(0)
    pose, _, _ = nodes.SAM3DBodyOpenPoseAnalyze().analyze(
        pose_image, device="cuda", inference_type="body", auto_head_proportion=False,
    )
    people = pose[0]["people"]
    summary = {"image": str(args.image), "size": [rgb.shape[1], rgb.shape[0]],
               "person_count": len(people), "people": []}
    for index, person in enumerate(people):
        body = person["pose_keypoints_2d"]
        head = nodes._pose_person_head_anchor(person)
        summary["people"].append({
            "index": index,
            "head_anchor": head,
            "visible_body_points": sum(body[i + 2] > 0 for i in range(0, len(body), 3)),
            "face_visibility": person.get("sam31_face_visibility"),
            "visible_face_points": sum(person.get("face_keypoints_2d", [])[i + 2] > 0
                                       for i in range(0, len(person.get("face_keypoints_2d", [])), 3)),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
