import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_NODES_DIR = os.path.dirname(ROOT_DIR)
LOCAL_SAM3D_DIR = ROOT_DIR
OPENPOSE_UTIL_PATH = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-ultimate-openpose-editor", "util.py")

try:
    import folder_paths
    DEFAULT_MODELS_DIR = folder_paths.models_dir
except Exception:
    DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "models"))

DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "sam3dbody")

_SAM3D_MODEL_CACHE = {}
_OPENPOSE_UTIL = None

BODY_COCO18_FROM_MHR70 = [0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
RIGHT_HAND_OPENPOSE_FROM_MHR70 = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]
LEFT_HAND_OPENPOSE_FROM_MHR70 = [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58]
BODY_SHAPE_IMAGES_TYPE = "SAM3DBODY_SHAPE_IMAGES"
BODY_SHAPE_INPUT_TYPE = f"IMAGE,{BODY_SHAPE_IMAGES_TYPE}"


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_jsonable(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _image_batch_size(image):
    if isinstance(image, torch.Tensor) and image.ndim >= 4:
        return int(image.shape[0])
    arr = np.asarray(image)
    return int(arr.shape[0]) if arr.ndim >= 4 else 1


def _body_shape_frames(body_shape_images=None):
    """Accept one IMAGE, an IMAGE batch, or an appended collection."""
    if body_shape_images is None:
        sources = ()
    elif isinstance(body_shape_images, (list, tuple)):
        sources = body_shape_images
    else:
        sources = (body_shape_images,)
    for source in sources:
        for batch_index in range(_image_batch_size(source)):
            yield source, batch_index


def _body_shape_slots_for_pose(pose_outputs, pose_people, slots):
    """Keep reference positions fixed when one frame failed or fewer were supplied."""
    outputs, faces = [], []
    for index, pose_output in enumerate(pose_outputs):
        shape_output, shape_face = slots[index] if index < len(slots) else (None, None)
        outputs.append(shape_output if shape_output is not None else pose_output)
        faces.append(shape_face if shape_face is not None else pose_people[index])
    return outputs, faces


def _comfy_image_to_rgb(image, batch_index=0):
    if isinstance(image, torch.Tensor):
        arr = image[int(batch_index)].detach().cpu().numpy()
    else:
        value = np.asarray(image)
        arr = value[int(batch_index)] if value.ndim >= 4 else value
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _pose_person_head_anchor(person):
    body = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32).reshape(-1, 3)
    if len(body) < 18:
        return None
    points = body[[0, 14, 15, 16, 17]]
    points = points[(points[:, 2] > 0) & np.isfinite(points[:, :2]).all(axis=1), :2]
    if len(points) == 0:
        return None
    anchor = points.mean(axis=0)
    head_size = max((float(np.linalg.norm(a - b)) for a in points for b in points), default=0.0)
    neck = body[1]
    neck_distance = float(np.linalg.norm(anchor - neck[:2])) if neck[2] > 0 and np.isfinite(neck[:2]).all() else 0.0
    return float(anchor[0]), float(anchor[1]), max(36.0, head_size * 2.8, neck_distance * 1.35)


def _make_model_config(model_path, device):
    model_path = os.path.abspath(model_path)
    ckpt_path = os.path.join(model_path, "model.ckpt")
    mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")
    if not (os.path.exists(ckpt_path) and os.path.exists(mhr_path)):
        try:
            from .model_download import download
            os.makedirs(model_path, exist_ok=True)
            print("[SAM3DBodyOpenPose] SAM3DBody model not found locally. Downloading from HuggingFace...")
            download("snapshot", repo_id="jetjodh/sam-3d-body-dinov3", local_dir=model_path)
        except Exception as e:
            raise RuntimeError(f"SAM3DBody model download failed. Place model.ckpt and assets/mhr_model.pt under {model_path}. Error: {e}") from e
    if device == "cuda" and not torch.cuda.is_available():
        print("[SAM3DBodyOpenPose] CUDA was selected but is not available. Falling back to CPU.")
        device = "cpu"
    return {"ckpt_path": ckpt_path, "mhr_path": mhr_path, "device": device, "model_path": model_path}


def _load_sam3d_model(model_config):
    if LOCAL_SAM3D_DIR not in sys.path:
        sys.path.insert(0, LOCAL_SAM3D_DIR)
    key = (model_config["ckpt_path"], model_config.get("device", "cuda"))
    if key in _SAM3D_MODEL_CACHE:
        return _SAM3D_MODEL_CACHE[key]
    from sam_3d_body.build_models import load_sam_3d_body
    model, model_cfg, mhr_path = load_sam_3d_body(model_config["ckpt_path"], device=model_config.get("device", "cuda"), mhr_path=model_config.get("mhr_path", ""))
    loaded = {"model": model, "model_cfg": model_cfg, "mhr_path": mhr_path, "device": model_config.get("device", "cuda")}
    _SAM3D_MODEL_CACHE[key] = loaded
    return loaded


def _flatten_keypoints(kpts_2d, mapping, confidence=1.0):
    out = []
    for src_idx in mapping:
        if src_idx < len(kpts_2d):
            x, y = kpts_2d[src_idx][:2]
            if np.isfinite(x) and np.isfinite(y):
                out.extend([float(x), float(y), float(confidence)])
                continue
        out.extend([0.0, 0.0, 0.0])
    return out


def _make_openpose_person(output, confidence=1.0):
    kpts = _to_numpy(output.get("pred_keypoints_2d"))
    if kpts is None or len(kpts) < 18:
        raise RuntimeError("SAM3DBody output did not include enough 2D keypoints.")
    return {
        "pose_keypoints_2d": _flatten_keypoints(kpts, BODY_COCO18_FROM_MHR70, confidence),
        "face_keypoints_2d": [],
        "hand_left_keypoints_2d": _flatten_keypoints(kpts, LEFT_HAND_OPENPOSE_FROM_MHR70, confidence),
        "hand_right_keypoints_2d": _flatten_keypoints(kpts, RIGHT_HAND_OPENPOSE_FROM_MHR70, confidence),
    }


def _make_sam3dbody_metadata(output):
    return {
        "format": "mhr70",
        "keypoints_2d": _to_jsonable(output.get("pred_keypoints_2d")),
        "keypoints_3d": _to_jsonable(output.get("pred_keypoints_3d")),
        "joint_coords": _to_jsonable(output.get("pred_joint_coords")),
        "joint_rotations": _to_jsonable(output.get("pred_global_rots")),
        "camera": _to_jsonable(output.get("pred_cam_t")),
        "focal_length": _to_jsonable(output.get("focal_length")),
        "pose_params": {
            "body_pose": _to_jsonable(output.get("body_pose_params")),
            "hand_pose": _to_jsonable(output.get("hand_pose_params")),
            "global_rot": _to_jsonable(output.get("global_rot")),
            "shape": _to_jsonable(output.get("shape_params")),
            "scale": _to_jsonable(output.get("scale_params")),
            "expr": _to_jsonable(output.get("expr_params")),
        },
    }


def _output_bbox(output):
    bbox = output.get("bbox") if isinstance(output, dict) else None
    if bbox is None:
        return None
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if len(arr) < 4 or not np.isfinite(arr[:4]).all():
        return None
    x1, y1, x2, y2 = [float(v) for v in arr[:4]]
    return {"min_x": min(x1, x2), "min_y": min(y1, y2), "width": abs(x2 - x1), "height": abs(y2 - y1)}


def _person_boxes_from_outputs(outputs):
    boxes = []
    for output in outputs:
        bbox = _output_bbox(output)
        boxes.append(None if bbox is None else [bbox["min_x"], bbox["min_y"],
                                                bbox["min_x"] + bbox["width"], bbox["min_y"] + bbox["height"]])
    return boxes


def _sort_outputs_top_left(outputs):
    indexed = []
    for i, output in enumerate(outputs or []):
        bbox = _output_bbox(output)
        if bbox is None:
            indexed.append((float("inf"), float("inf"), i, output))
        else:
            row_h = max(1.0, bbox["height"] * 0.5)
            indexed.append((round(bbox["min_y"] / row_h), bbox["min_x"], i, output))
    return [item[3] for item in sorted(indexed, key=lambda x: (x[0], x[1], x[2]))]


def _make_openpose_keypoint_from_outputs(outputs, width, height, confidence=1.0):
    return {"people": [_make_openpose_person(o, confidence) for o in outputs], "canvas_width": int(width), "canvas_height": int(height)}


def _make_body_shape_keypoint_from_outputs(outputs, width, height, confidence=1.0):
    pose = _make_openpose_keypoint_from_outputs(outputs, width, height, confidence)
    pose["sam3dbody_people"] = [_make_sam3dbody_metadata(o) for o in outputs]
    if pose["sam3dbody_people"]:
        pose["sam3dbody"] = pose["sam3dbody_people"][0]
    return pose


def _run_sam3d_estimator(estimator, image, bbox_threshold, inference_type, batch_index=0, person_threshold=0.5):
    if __package__:
        from .sam31_face import detect_people
    else:
        from sam31_face import detect_people

    img_rgb = _comfy_image_to_rgb(image, batch_index)
    height, width = img_rgb.shape[:2]
    people = detect_people(img_rgb, threshold=person_threshold)
    if not people:
        raise RuntimeError("SAM3.1 did not detect any people in the image")
    outputs, width, height = _run_sam3d_estimator_with_regions(
        estimator, image, people, bbox_threshold, inference_type, batch_index)
    return _sort_outputs_top_left(outputs), width, height, people


def _run_sam3d_estimator_with_regions(estimator, image, people, bbox_threshold, inference_type, batch_index=0):
    img_rgb = _comfy_image_to_rgb(image, batch_index)
    height, width = img_rgb.shape[:2]
    if not people:
        return [], width, height
    bboxes = np.asarray([person["box"] for person in people], dtype=np.float32)
    masks = np.asarray([person["mask"] for person in people], dtype=np.uint8)
    print(f"[SAM3DBodyOpenPose] Using SAM3.1 person boxes and masks: {len(people)}")
    outputs = estimator.process_one_image(img_rgb, bboxes=bboxes, masks=masks,
                                          bbox_thr=bbox_threshold, inference_type=inference_type)
    if not outputs:
        raise RuntimeError("SAM3DBody could not estimate any SAM3.1 person regions")
    return outputs, width, height


def _debug_crops_to_image_tensor(_crops):
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _pose_canvas_size(pose_keypoint):
    first = pose_keypoint[0] if isinstance(pose_keypoint, list) and pose_keypoint else pose_keypoint
    if isinstance(first, dict):
        return int(first.get("canvas_width", 512) or 512), int(first.get("canvas_height", 768) or 768)
    return 512, 768


def _openpose_util():
    global _OPENPOSE_UTIL
    if _OPENPOSE_UTIL is not None:
        return _OPENPOSE_UTIL
    if not os.path.exists(OPENPOSE_UTIL_PATH):
        raise RuntimeError("ComfyUI-ultimate-openpose-editor util.py was not found.")
    spec = importlib.util.spec_from_file_location("sam3dbody_openpose_uope_util", OPENPOSE_UTIL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _OPENPOSE_UTIL = module
    return module


class SAM3DBodyOpenPoseBodyShapeAppend:
    """Append one image or image batch to a chain of body-shape references."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"body_shape_image": ("IMAGE",)},
                "optional": {"body_shape_images": (BODY_SHAPE_IMAGES_TYPE,)}}

    RETURN_TYPES = (BODY_SHAPE_IMAGES_TYPE,)
    RETURN_NAMES = ("body_shape_images",)
    FUNCTION = "append"
    CATEGORY = "SAM3DBody/openpose"

    def append(self, body_shape_image, body_shape_images=None):
        return (tuple(body_shape_images or ()) + (body_shape_image,),)


class SAM3DBodyOpenPoseAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_image": ("IMAGE",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "bbox_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "inference_type": (["full", "body", "hand"], {"default": "full"}),
                "confidence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "auto_head_proportion": ("BOOLEAN", {"default": True}),
                "person_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {"body_shape_images": (BODY_SHAPE_INPUT_TYPE,)},
        }
    RETURN_TYPES = ("POSE_KEYPOINT", "POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES = ("pose_keypoint", "body_shape_keypoint", "debug_person_crops")
    FUNCTION = "analyze"
    CATEGORY = "SAM3DBody/openpose"

    def analyze(self, pose_image, device="cuda", bbox_threshold=0.8, inference_type="full", confidence=1.0, auto_head_proportion=True, person_threshold=0.5, body_shape_images=None):
        model_config = _make_model_config(DEFAULT_MODEL_PATH, device)
        loaded = _load_sam3d_model(model_config)
        from sam_3d_body import SAM3DBodyEstimator
        estimator = SAM3DBodyEstimator(loaded["model"], loaded["model_cfg"], human_detector=None, human_segmentor=None, fov_estimator=None)
        pose_outputs, pose_width, pose_height, pose_people = _run_sam3d_estimator(
            estimator, pose_image, bbox_threshold, inference_type, 0, person_threshold)
        pose_pose = _make_openpose_keypoint_from_outputs(pose_outputs, pose_width, pose_height, confidence)
        try:
            from .sam31_face import detect_parts, update_pose
            anchors = [_pose_person_head_anchor(p) for p in pose_pose.get("people", [])]
            if anchors:
                rgb = _comfy_image_to_rgb(pose_image, 0)
                parts = detect_parts(rgb, anchors, person_boxes=_person_boxes_from_outputs(pose_outputs),
                                     person_candidates=pose_people)
                pose_pose = update_pose(pose_pose, parts)
        except Exception as e:
            print(f"[SAM3DBodyOpenPose] SAM3.1 face-part detection skipped. Error: {e}")
        body_shape_outputs = pose_outputs
        body_shape_face_people = []
        if body_shape_images is not None:
            slots = []
            for source, batch_index in _body_shape_frames(body_shape_images):
                if len(slots) >= len(pose_outputs):
                    break
                try:
                    outs, shape_width, shape_height, shape_people = _run_sam3d_estimator(
                        estimator, source, bbox_threshold, inference_type, batch_index, person_threshold)
                    face_people = [None] * len(outs)
                    if auto_head_proportion:
                        shape_pose = _make_openpose_keypoint_from_outputs(outs, shape_width, shape_height, confidence)
                        try:
                            from .sam31_face import detect_parts, update_pose
                            shape_anchors = [_pose_person_head_anchor(p) for p in shape_pose["people"]]
                            shape_parts = detect_parts(_comfy_image_to_rgb(source, batch_index), shape_anchors,
                                                       person_boxes=_person_boxes_from_outputs(outs),
                                                       person_candidates=shape_people)
                            shape_pose = update_pose(shape_pose, shape_parts)
                        except Exception as e:
                            print(f"[SAM3DBodyOpenPose] body-shape face detection skipped for batch {batch_index}. Error: {e}")
                        face_people = shape_pose["people"]
                    slots.extend(zip(outs, face_people))
                except Exception as e:
                    print(f"[SAM3DBodyOpenPose] body-shape reference batch {batch_index} skipped. Error: {e}")
                    slots.append((None, None))
            body_shape_outputs, body_shape_face_people = _body_shape_slots_for_pose(
                pose_outputs, pose_pose["people"], slots)
        body_shape_pose = _make_body_shape_keypoint_from_outputs(body_shape_outputs, pose_width, pose_height, confidence)
        if body_shape_images is not None and slots:
            if __package__:
                from .sam31_face import normalize_body_shape_to_pose
            else:
                from sam31_face import normalize_body_shape_to_pose
            body_shape_pose = normalize_body_shape_to_pose(
                body_shape_pose, pose_pose, body_shape_face_people if auto_head_proportion else None)
        return ([pose_pose], [body_shape_pose], _debug_crops_to_image_tensor([]))


class SAM3DBodyOpenPoseVideoAnalyze:
    """Analyze an IMAGE batch as ordered video frames with stable person slots."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = SAM3DBodyOpenPoseAnalyze.INPUT_TYPES()
        inputs["required"]["pose_image"] = ("IMAGE",)
        inputs["required"]["max_missing_frames"] = ("INT", {"default": 12, "min": 0, "max": 300})
        inputs["required"]["tracking_mode"] = (["sam31_video", "frame_matching"], {"default": "sam31_video"})
        inputs["required"]["max_tracked_people"] = ("INT", {"default": 16, "min": 1, "max": 64})
        inputs["required"]["detect_interval"] = ("INT", {"default": 1, "min": 1, "max": 60})
        inputs["optional"]["track_data"] = ("SAM3_TRACK_DATA",)
        return inputs

    RETURN_TYPES = ("POSE_KEYPOINT", "POSE_KEYPOINT")
    RETURN_NAMES = ("pose_keypoint", "body_shape_keypoint")
    FUNCTION = "analyze_video"
    CATEGORY = "SAM3DBody/openpose"

    def analyze_video(self, pose_image, device="cuda", bbox_threshold=0.8, inference_type="full",
                      confidence=1.0, auto_head_proportion=True, person_threshold=0.5,
                      max_missing_frames=12, tracking_mode="sam31_video", max_tracked_people=16,
                      detect_interval=1, body_shape_images=None, track_data=None):
        if __package__:
            from .sam31_face import detect_parts, update_pose, normalize_body_shape_to_pose
            from .sam31_video import track_people_video
            from .video_tracking import PersonTracker
        else:
            from sam31_face import detect_parts, update_pose, normalize_body_shape_to_pose
            from sam31_video import track_people_video
            from video_tracking import PersonTracker
        if tracking_mode not in ("sam31_video", "frame_matching"):
            raise ValueError(f"Unknown tracking_mode: {tracking_mode}")
        use_native_tracking = track_data is not None or tracking_mode == "sam31_video"
        tracked_frames = (track_people_video(pose_image, person_threshold, max_tracked_people,
                                             detect_interval, track_data) if use_native_tracking else None)
        model_config = _make_model_config(DEFAULT_MODEL_PATH, device)
        loaded = _load_sam3d_model(model_config)
        from sam_3d_body import SAM3DBodyEstimator
        estimator = SAM3DBodyEstimator(loaded["model"], loaded["model_cfg"],
                                      human_detector=None, human_segmentor=None, fov_estimator=None)
        slots = []
        for source, batch_index in _body_shape_frames(body_shape_images):
            try:
                outs, shape_width, shape_height, shape_people = _run_sam3d_estimator(
                    estimator, source, bbox_threshold, inference_type, batch_index, person_threshold)
                face_people = [None] * len(outs)
                if auto_head_proportion:
                    shape_pose = _make_openpose_keypoint_from_outputs(outs, shape_width, shape_height, confidence)
                    try:
                        anchors = [_pose_person_head_anchor(p) for p in shape_pose["people"]]
                        parts = detect_parts(_comfy_image_to_rgb(source, batch_index), anchors,
                                             person_boxes=_person_boxes_from_outputs(outs),
                                             person_candidates=shape_people)
                        face_people = update_pose(shape_pose, parts)["people"]
                    except Exception as e:
                        print(f"[SAM3DBodyOpenPose] video body-shape face detection skipped for batch {batch_index}. Error: {e}")
                slots.extend(zip(outs, face_people))
            except Exception as e:
                print(f"[SAM3DBodyOpenPose] video body-shape reference batch {batch_index} skipped. Error: {e}")
                slots.append((None, None))

        tracker = PersonTracker(max_missing_frames)
        pose_frames, shape_frames = [], []
        for frame_index in range(_image_batch_size(pose_image)):
            rgb = _comfy_image_to_rgb(pose_image, frame_index)
            height, width = rgb.shape[:2]
            if use_native_tracking:
                people = tracked_frames[frame_index]
                ids = [person["track_id"] for person in people]
                outputs, width, height = _run_sam3d_estimator_with_regions(
                    estimator, pose_image, people, bbox_threshold, inference_type, frame_index)
                if len(outputs) != len(ids):
                    raise RuntimeError(f"SAM3DBody returned {len(outputs)} people for {len(ids)} SAM3.1 tracks in frame {frame_index}")
            else:
                try:
                    outputs, width, height, people = _run_sam3d_estimator(
                        estimator, pose_image, bbox_threshold, inference_type, frame_index, person_threshold)
                except RuntimeError as e:
                    if "did not detect any people" not in str(e) and "could not estimate any SAM3.1 person regions" not in str(e):
                        raise
                    outputs, people = [], []
                ids = tracker.update(rgb, outputs, people, frame_index)
                order = sorted(range(len(ids)), key=lambda i: ids[i])
                outputs = [outputs[i] for i in order]
                ids = [ids[i] for i in order]
            if not outputs:
                print(f"[SAM3DBodyOpenPose] video frame {frame_index}: no people")
                pose_frames.append({"people": [], "canvas_width": width, "canvas_height": height})
                shape_frames.append({"people": [], "canvas_width": width, "canvas_height": height})
                continue
            pose_pose = _make_openpose_keypoint_from_outputs(outputs, width, height, confidence)
            try:
                anchors = [_pose_person_head_anchor(p) for p in pose_pose["people"]]
                if anchors:
                    parts = detect_parts(rgb, anchors, person_boxes=_person_boxes_from_outputs(outputs),
                                         person_candidates=people)
                    pose_pose = update_pose(pose_pose, parts)
            except Exception as e:
                print(f"[SAM3DBodyOpenPose] video frame {frame_index}: face-part detection skipped. Error: {e}")

            shape_outputs, shape_face_people = [], []
            for index, id_ in enumerate(ids):
                shape_output, shape_face = slots[id_] if id_ < len(slots) else (None, None)
                shape_outputs.append(shape_output if shape_output is not None else outputs[index])
                shape_face_people.append(shape_face if shape_face is not None else pose_pose["people"][index])
            shape_pose = _make_body_shape_keypoint_from_outputs(shape_outputs, width, height, confidence)
            if slots:
                shape_pose = normalize_body_shape_to_pose(
                    shape_pose, pose_pose, shape_face_people if auto_head_proportion else None)
            for person, id_ in zip(pose_pose["people"], ids):
                person["track_id"] = id_
            for person, id_ in zip(shape_pose["people"], ids):
                person["track_id"] = id_
            pose_frames.append(pose_pose)
            shape_frames.append(shape_pose)
        return (pose_frames, shape_frames)


class SAM3DBodyOpenPoseRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose_keypoint": ("POSE_KEYPOINT",), "show_body": ("BOOLEAN", {"default": True}), "show_face": ("BOOLEAN", {"default": True}), "show_hands": ("BOOLEAN", {"default": True}), "width": ("INT", {"default": 0, "min": 0, "max": 12800}), "height": ("INT", {"default": 0, "min": 0, "max": 12800}), "offset_x": ("FLOAT", {"default": 0.0}), "offset_y": ("FLOAT", {"default": 0.0}), "use_ground_plane": ("BOOLEAN", {"default": True}), "pose_marker_size": ("INT", {"default": 4, "min": 0, "max": 100}), "face_marker_size": ("INT", {"default": 3, "min": 0, "max": 100}), "hand_marker_size": ("INT", {"default": 2, "min": 0, "max": 100}), "pelvis_scale": ("FLOAT", {"default": 1.0}), "torso_scale": ("FLOAT", {"default": 1.0}), "neck_scale": ("FLOAT", {"default": 1.0}), "head_scale": ("FLOAT", {"default": 1.0}), "eye_distance_scale": ("FLOAT", {"default": 1.0}), "left_eye_scale": ("FLOAT", {"default": 1.0}), "right_eye_scale": ("FLOAT", {"default": 1.0}), "left_eyebrow_scale": ("FLOAT", {"default": 1.0}), "right_eyebrow_scale": ("FLOAT", {"default": 1.0}), "mouth_scale": ("FLOAT", {"default": 1.0}), "nose_scale_face": ("FLOAT", {"default": 1.0}), "face_shape_scale": ("FLOAT", {"default": 1.0}), "shoulder_scale": ("FLOAT", {"default": 1.0}), "arm_scale": ("FLOAT", {"default": 1.0}), "leg_scale": ("FLOAT", {"default": 1.0}), "hands_scale": ("FLOAT", {"default": 1.0}), "overall_scale": ("FLOAT", {"default": 1.0})}, "optional": {"body_shape_keypoint": ("POSE_KEYPOINT",)}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("openpose_image",)
    FUNCTION = "render"
    CATEGORY = "SAM3DBody/openpose"

    def render(self, pose_keypoint, show_body=True, show_face=True, show_hands=True, width=0, height=0, offset_x=0.0, offset_y=0.0, use_ground_plane=True, pose_marker_size=4, face_marker_size=3, hand_marker_size=2, pelvis_scale=1.0, torso_scale=1.0, neck_scale=1.0, head_scale=1.0, eye_distance_scale=1.0, left_eye_scale=1.0, right_eye_scale=1.0, left_eyebrow_scale=1.0, right_eyebrow_scale=1.0, mouth_scale=1.0, nose_scale_face=1.0, face_shape_scale=1.0, shoulder_scale=1.0, arm_scale=1.0, leg_scale=1.0, hands_scale=1.0, overall_scale=1.0, body_shape_keypoint=None):
        util = _openpose_util()
        source_width, source_height = _pose_canvas_size(pose_keypoint)
        render_width = int(width) if int(width) > 0 else source_width
        render_height = int(height) if int(height) > 0 else source_height
        frames = pose_keypoint if isinstance(pose_keypoint, list) else [pose_keypoint]
        draw_frames = []
        for frame in frames:
            if not isinstance(frame, dict):
                draw_frames.append(frame)
                continue
            draw_frame = dict(frame)
            draw_frame["people"] = [
                {**person, "face_keypoints_2d": [] if len(person.get("face_keypoints_2d", [])) == 210 else person.get("face_keypoints_2d", [])}
                for person in frame.get("people", [])
            ]
            draw_frames.append(draw_frame)
        def draw(frames_to_draw, target):
            return util.draw_pose_json(json.dumps(frames_to_draw), render_width, use_ground_plane,
                                       show_body, show_face, show_hands, pose_marker_size, face_marker_size,
                                       hand_marker_size, pelvis_scale, torso_scale, neck_scale, head_scale,
                                       eye_distance_scale, 0.0, 0.0, left_eye_scale, right_eye_scale,
                                       left_eyebrow_scale, right_eyebrow_scale, mouth_scale, nose_scale_face,
                                       face_shape_scale, shoulder_scale, arm_scale, leg_scale, hands_scale,
                                       overall_scale, 0.0, offset_x, offset_y,
                                       target_pose_keypoint_obj=target)[0]

        if body_shape_keypoint is None:
            pose_imgs = draw(draw_frames, None)
        else:
            target_frames = body_shape_keypoint if isinstance(body_shape_keypoint, list) else [body_shape_keypoint]
            pose_imgs = []
            for frame_index, frame in enumerate(draw_frames):
                if not isinstance(frame, dict):
                    pose_imgs.extend(draw([frame], None))
                    continue
                target_frame = target_frames[min(frame_index, len(target_frames) - 1)] if target_frames else None
                target_people = target_frame.get("people", []) if isinstance(target_frame, dict) else []
                canvas = None
                for person_index, person in enumerate(frame.get("people", [])):
                    single_frame = {**frame, "people": [person]}
                    single_target = ([{**target_frame, "people": [target_people[person_index]]}]
                                     if person_index < len(target_people) else None)
                    images = draw([single_frame], single_target)
                    if images:
                        canvas = images[0] if canvas is None else np.maximum(canvas, images[0])
                pose_imgs.append(canvas if canvas is not None else np.zeros((render_height, render_width, 3), dtype=np.uint8))
        pose_imgs = pose_imgs or [np.zeros((render_height, render_width, 3), dtype=np.uint8)]
        if render_height > 0:
            import cv2
            pose_imgs = [cv2.resize(img, (render_width, render_height), interpolation=cv2.INTER_AREA) if img.shape[0] != render_height or img.shape[1] != render_width else img for img in pose_imgs]
        return (torch.from_numpy(np.asarray(pose_imgs).astype(np.float32) / 255.0),)


NODE_CLASS_MAPPINGS = {"SAM3DBodyOpenPoseAnalyze": SAM3DBodyOpenPoseAnalyze,
                       "SAM3DBodyOpenPoseVideoAnalyze": SAM3DBodyOpenPoseVideoAnalyze,
                       "SAM3DBodyOpenPoseBodyShapeAppend": SAM3DBodyOpenPoseBodyShapeAppend,
                       "SAM3DBodyOpenPoseRender": SAM3DBodyOpenPoseRender}
NODE_DISPLAY_NAME_MAPPINGS = {"SAM3DBodyOpenPoseAnalyze": "SAM3D Body OpenPose Analyze",
                              "SAM3DBodyOpenPoseVideoAnalyze": "SAM3D Body OpenPose Video Analyze",
                              "SAM3DBodyOpenPoseBodyShapeAppend": "SAM3D Body Shape Images (Append)",
                              "SAM3DBodyOpenPoseRender": "SAM3D Body OpenPose Render"}
