"""Compare legacy guided boxes with SAM3.1 person boxes and masks on one image."""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _summary(nodes, outputs, width, height, duration):
    pose = nodes._make_openpose_keypoint_from_outputs(outputs, width, height)
    people = []
    for index, (output, person) in enumerate(zip(outputs, pose["people"])):
        body = person["pose_keypoints_2d"]
        people.append({"index": index, "bbox": nodes._output_bbox(output),
                       "head_anchor": nodes._pose_person_head_anchor(person),
                       "visible_body_points": sum(body[i + 2] > 0 for i in range(0, len(body), 3))})
    return {"seconds": round(duration, 2), "person_count": len(people), "people": people}, pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("validation"))
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project.parents[1]))
    sys.path.insert(0, str(project.parent))
    sys.argv = [sys.argv[0], "--disable-xformers"]
    import comfy.options
    comfy.options.enable_args_parsing()

    nodes = importlib.import_module(f"{project.name}.nodes")
    sam31 = importlib.import_module(f"{project.name}.sam31_face")

    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
    config = nodes._make_model_config(nodes.DEFAULT_MODEL_PATH, "cuda")
    loaded = nodes._load_sam3d_model(config)
    from sam_3d_body import SAM3DBodyEstimator
    estimator = SAM3DBodyEstimator(loaded["model"], loaded["model_cfg"],
                                   human_detector=None, human_segmentor=None, fov_estimator=None)

    started = time.perf_counter()
    old_outputs, _, _ = nodes._run_sam3d_estimator(estimator, tensor, .8, "body", 0, .25, "cuda")
    old_result, old_pose = _summary(nodes, old_outputs, width, height, time.perf_counter() - started)

    started = time.perf_counter()
    candidates = sam31.detect_people(rgb)
    boxes = np.asarray([person["box"] for person in candidates], dtype=np.float32)
    masks = np.asarray([person["mask"] for person in candidates], dtype=np.uint8)
    if len(candidates):
        new_outputs = estimator.process_one_image(rgb, bboxes=boxes, masks=masks, inference_type="body")
        new_outputs = nodes._sort_outputs_top_left(new_outputs)
        new_result, new_pose = _summary(nodes, new_outputs, width, height, time.perf_counter() - started)
    else:
        new_result, new_pose = {"seconds": round(time.perf_counter() - started, 2), "person_count": 0,
                                "people": []}, None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, pose in (("traditional", old_pose), ("sam31_masks", new_pose)):
        if pose is None:
            continue
        try:
            rendered, = nodes.SAM3DBodyOpenPoseRender().render([pose], width=width, height=height)
            canvas = (rendered[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(canvas).save(args.output_dir / f"{name}_pose.png")
        except Exception as error:
            print(f"[validation] {name} render unavailable: {error}")
    result = {"image": str(args.image), "size": [width, height],
              "sam31_candidates": len(candidates), "traditional": old_result, "sam31_masks": new_result}
    (args.output_dir / "person_input_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
