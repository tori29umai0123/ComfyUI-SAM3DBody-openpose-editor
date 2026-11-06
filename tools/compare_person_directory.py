"""Compare guided boxes and SAM3.1 person masks across an image directory."""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from compare_person_inputs import _summary


def _render(nodes, pose, width, height, path):
    if pose is None:
        return
    rendered, = nodes.SAM3DBodyOpenPoseRender().render([pose], width=width, height=height)
    Image.fromarray((rendered[0].cpu().numpy() * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("validation/all_images"))
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project.parents[1]))
    sys.path.insert(0, str(project.parent))
    sys.argv = [sys.argv[0], "--disable-xformers"]
    import comfy.options
    comfy.options.enable_args_parsing()

    nodes = importlib.import_module(f"{project.name}.nodes")
    sam31 = importlib.import_module(f"{project.name}.sam31_face")
    config = nodes._make_model_config(nodes.DEFAULT_MODEL_PATH, "cuda")
    loaded = nodes._load_sam3d_model(config)
    from sam_3d_body import SAM3DBodyEstimator
    estimator = SAM3DBodyEstimator(loaded["model"], loaded["model_cfg"],
                                   human_detector=None, human_segmentor=None, fov_estimator=None)
    images = sorted(path for path in args.directory.iterdir()
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(images, 1):
        print(f"[validation] {index}/{len(images)}: {path.name}", flush=True)
        record = {"image": str(path), "name": path.name}
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            height, width = rgb.shape[:2]
            record["size"] = [width, height]
            tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
            started = time.perf_counter()
            old_outputs, _, _ = nodes._run_sam3d_estimator(estimator, tensor, .8, "body", 0, .25, "cuda")
            old_result, old_pose = _summary(nodes, old_outputs, width, height, time.perf_counter() - started)

            started = time.perf_counter()
            candidates = sam31.detect_people(rgb)
            boxes = np.asarray([person["box"] for person in candidates], dtype=np.float32)
            masks = np.asarray([person["mask"] for person in candidates], dtype=np.uint8)
            if candidates:
                new_outputs = estimator.process_one_image(rgb, bboxes=boxes, masks=masks, inference_type="body")
                new_outputs = nodes._sort_outputs_top_left(new_outputs)
                new_result, new_pose = _summary(nodes, new_outputs, width, height, time.perf_counter() - started)
            else:
                new_result = {"seconds": round(time.perf_counter() - started, 2), "person_count": 0, "people": []}
                new_pose = None
            record.update({"sam31_candidates": len(candidates), "traditional": old_result,
                           "sam31_masks": new_result})
            stem = f"{index:02d}_{path.stem}"
            _render(nodes, old_pose, width, height, args.output_dir / f"{stem}_traditional.png")
            _render(nodes, new_pose, width, height, args.output_dir / f"{stem}_sam31.png")
            print(f"[validation] people: traditional={old_result['person_count']}, "
                  f"sam31={new_result['person_count']}", flush=True)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"[validation] error: {record['error']}", flush=True)
        records.append(record)
        (args.output_dir / "batch_comparison.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[validation] saved {len(records)} records to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
