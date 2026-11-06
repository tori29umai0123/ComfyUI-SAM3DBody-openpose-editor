import importlib.util
import json
import os
import shutil
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
ANIMEHEADS_MODEL_RELATIVE_PATH = os.path.join("animeheads", "animeheadsv3.onnx")
RTDETRV4_BODY_MODEL_RELATIVE_PATH = os.path.join("rtdetrv4", "rtdetrv4-x-manga109s_v2.onnx")
RTDETRV4_BODY_REPO_ID = "tori29umai/rtdetrv4-x-manga109s_v2"
RTDETRV4_BODY_REPO_FILENAME = "model.onnx"
RTDETRV4_BODY_SCORE_THRESHOLD = 0.10

_SAM3D_MODEL_CACHE = {}
_ANIMEHEADS_NET_CACHE = {}
_RTDETRV4_BODY_SESSION_CACHE = {}
_ANIME_FACE_DETECTOR_CACHE = {}
_ANIME_FACE_DETECTOR_WARNED = False
_OPENPOSE_UTIL = None

BODY_COCO18_FROM_MHR70 = [0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
RIGHT_HAND_OPENPOSE_FROM_MHR70 = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]
LEFT_HAND_OPENPOSE_FROM_MHR70 = [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58]


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


def _comfy_image_to_rgb(image, batch_index=0):
    if isinstance(image, torch.Tensor):
        arr = image[int(batch_index)].detach().cpu().numpy()
    else:
        value = np.asarray(image)
        arr = value[int(batch_index)] if value.ndim >= 4 else value
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _comfy_image_to_bgr(image, batch_index=0):
    return _comfy_image_to_rgb(image, batch_index)[..., ::-1].copy()


def _resolve_model_path_under_models(model_path, default_relative_path):
    if not model_path:
        model_path = default_relative_path
    model_path = os.path.expanduser(str(model_path))
    if not os.path.isabs(model_path):
        model_path = os.path.join(DEFAULT_MODELS_DIR, model_path)
    return os.path.abspath(model_path)


def _ensure_animeheads_model(model_path=None):
    model_path = _resolve_model_path_under_models(model_path, ANIMEHEADS_MODEL_RELATIVE_PATH)
    if os.path.exists(model_path):
        return model_path
    try:
        from .model_download import download
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("[SAM3DBodyOpenPose] AnimeHeads model not found locally. Downloading from HuggingFace...")
        downloaded = download("file", repo_id="tori29umai/AnimeHeads-ONNX", filename="weights/animeheadsv3.onnx", repo_type="model")
        shutil.copyfile(downloaded, model_path)
        return model_path
    except Exception as e:
        raise RuntimeError(f"AnimeHeads ONNX download failed. Place it at {model_path}. Error: {e}") from e


def _ensure_rtdetrv4_body_model(model_path=None):
    model_path = _resolve_model_path_under_models(model_path, RTDETRV4_BODY_MODEL_RELATIVE_PATH)
    if os.path.exists(model_path):
        return model_path
    try:
        from .model_download import download
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("[SAM3DBodyOpenPose] RT-DETRv4 body model not found locally. Downloading from HuggingFace...")
        downloaded = download("file", repo_id=RTDETRV4_BODY_REPO_ID, filename=RTDETRV4_BODY_REPO_FILENAME, repo_type="model")
        shutil.copyfile(downloaded, model_path)
        return model_path
    except Exception as e:
        raise RuntimeError(f"RT-DETRv4 body ONNX download failed. Place it at {model_path}. Error: {e}") from e


def _load_animeheads_net(model_path=None):
    import cv2
    model_path = _ensure_animeheads_model(model_path)
    key = os.path.abspath(model_path)
    if key not in _ANIMEHEADS_NET_CACHE:
        _ANIMEHEADS_NET_CACHE[key] = cv2.dnn.readNetFromONNX(key)
    return _ANIMEHEADS_NET_CACHE[key]


def _load_rtdetrv4_body_session(model_path=None):
    model_path = _ensure_rtdetrv4_body_model(model_path)
    key = os.path.abspath(model_path)
    if key not in _RTDETRV4_BODY_SESSION_CACHE:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        _RTDETRV4_BODY_SESSION_CACHE[key] = ort.InferenceSession(key, providers=providers)
    return _RTDETRV4_BODY_SESSION_CACHE[key]


def _letterbox_bgr(image, size=640):
    import cv2
    h, w = image.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, px, py


def _detect_anime_heads(image_bgr, model_path=None, confidence=0.25):
    import cv2
    if image_bgr is None:
        return []
    net = _load_animeheads_net(model_path)
    inp, scale, px, py = _letterbox_bgr(image_bgr, 640)
    blob = cv2.dnn.blobFromImage(inp, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] == 5:
        out = out.T
    ih, iw = image_bgr.shape[:2]
    boxes, scores = [], []
    for row in out:
        if len(row) < 5:
            continue
        cx, cy, w, h, score = [float(v) for v in row[:5]]
        if score < float(confidence):
            continue
        x1 = max(0.0, min(float(iw), (cx - w * 0.5 - px) / scale))
        y1 = max(0.0, min(float(ih), (cy - h * 0.5 - py) / scale))
        x2 = max(0.0, min(float(iw), (cx + w * 0.5 - px) / scale))
        y2 = max(0.0, min(float(ih), (cy + h * 0.5 - py) / scale))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))])
        scores.append(score)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=float(confidence), nms_threshold=0.45)
    if len(keep) == 0:
        return []
    detections = []
    for i in np.array(keep).reshape(-1).tolist():
        x, y, w, h = boxes[i]
        detections.append({"bbox": [float(x), float(y), float(x + w), float(y + h)], "confidence": float(scores[i])})
    detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return detections


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, aa + ba - inter)


def _nms_dict_boxes(items, iou_thr):
    items = sorted(items or [], key=lambda x: float(x.get("score", x.get("confidence", 0.0))), reverse=True)
    kept = []
    for item in items:
        if all(_bbox_iou(item["bbox"], k["bbox"]) <= iou_thr for k in kept):
            kept.append(item)
    return kept


def _detect_rtdetrv4_body_persons(image_bgr, score_thr=RTDETRV4_BODY_SCORE_THRESHOLD):
    session = _load_rtdetrv4_body_session()
    image_rgb = image_bgr[..., ::-1].copy()
    labels, boxes, scores = session.run(None, {"image": image_rgb[None, ...]})
    labels = np.asarray(labels[0])
    boxes = np.asarray(boxes[0], dtype=np.float32)
    scores = np.asarray(scores[0], dtype=np.float32)
    keep = (labels == 0) & (scores >= float(score_thr))
    out = []
    for box, score in zip(boxes[keep], scores[keep]):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if x2 > x1 and y2 > y1:
            out.append({"bbox": [x1, y1, x2, y2], "score": float(score)})
    return _nms_dict_boxes(out, 0.55)


def _face_bbox_center(face):
    bbox = face.get("bbox", []) if isinstance(face, dict) else []
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, max(abs(x2 - x1), abs(y2 - y1))


def _target_box_assignment_score(head, person_box):
    center = _face_bbox_center(head)
    if center is None or person_box is None:
        return -999.0
    fx, fy, fsize = center
    x1, y1, x2, y2 = [float(v) for v in person_box[:4]]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    inside = 1.0 if x1 <= fx <= x2 and y1 <= fy <= y2 else 0.0
    dist = np.hypot((fx - (x1 + x2) * 0.5) / bw, (fy - (y1 + y2) * 0.5) / bh)
    return float(inside * 4.0 + min(1.0, fsize / max(1.0, min(bw, bh))) - dist)


def _body_boxes_from_heads_and_rtdetr(image_bgr, head_confidence, device):
    heads = _detect_anime_heads(image_bgr, ANIMEHEADS_MODEL_RELATIVE_PATH, head_confidence) or []
    if not heads:
        return None, []
    for i, h in enumerate(heads):
        h["face_id"] = i
    try:
        persons = _detect_rtdetrv4_body_persons(image_bgr)
    except Exception as e:
        print(f"[SAM3DBodyOpenPose] RT-DETRv4 body boxes skipped. Error: {e}")
        persons = []
    used, boxes = set(), []
    ih, iw = image_bgr.shape[:2]
    for head in heads:
        best_i, best_s = None, -999.0
        for idx, person in enumerate(persons):
            if idx in used:
                continue
            score = _target_box_assignment_score(head, person.get("bbox"))
            if score > best_s:
                best_i, best_s = idx, score
        if best_i is not None and best_s >= -0.15:
            used.add(best_i)
            box = persons[best_i]["bbox"]
        else:
            x1, y1, x2, y2 = head["bbox"]
            hw, hh = x2 - x1, y2 - y1
            cx = (x1 + x2) * 0.5
            box = [cx - hw * 2.4, y1 - hh * 0.9, cx + hw * 2.4, y2 + hh * 6.0]
            box = [max(0.0, box[0]), max(0.0, box[1]), min(float(iw), box[2]), min(float(ih), box[3])]
        boxes.append(box)
    return np.asarray(boxes, dtype=np.float32), heads


def _anime_face_device(device):
    return "cuda:0" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"


def _load_anime_face_detector(device="cuda"):
    global _ANIME_FACE_DETECTOR_WARNED
    key = ("yolov3", _anime_face_device(device))
    if key in _ANIME_FACE_DETECTOR_CACHE:
        return _ANIME_FACE_DETECTOR_CACHE[key]
    try:
        from anime_face_detector import create_detector
        detector = create_detector("yolov3", device=key[1])
        _ANIME_FACE_DETECTOR_CACHE[key] = detector
        return detector
    except Exception as e:
        if not _ANIME_FACE_DETECTOR_WARNED:
            print(f"[SAM3DBodyOpenPose] anime-face-detector unavailable; face keypoints skipped. Error: {e}")
            _ANIME_FACE_DETECTOR_WARNED = True
        _ANIME_FACE_DETECTOR_CACHE[key] = None
        return None


def _anime_face_from_prediction(pred, offset_x=0.0, offset_y=0.0):
    bbox = _to_numpy(pred.get("bbox")) if isinstance(pred, dict) else None
    keypoints = _to_numpy(pred.get("keypoints")) if isinstance(pred, dict) else None
    if bbox is None or keypoints is None:
        return None
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    if bbox.ndim != 1 or len(bbox) < 4 or keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return None
    bbox[0] += float(offset_x); bbox[2] += float(offset_x)
    bbox[1] += float(offset_y); bbox[3] += float(offset_y)
    keypoints[:, 0] += float(offset_x); keypoints[:, 1] += float(offset_y)
    return {"bbox": _to_jsonable(bbox), "keypoints": _to_jsonable(keypoints[:, :3] if keypoints.shape[1] >= 3 else keypoints[:, :2])}


def _append_unique_face(faces, face):
    center = _face_bbox_center(face)
    if center is None:
        return
    cx, cy, size = center
    for existing in faces:
        other = _face_bbox_center(existing)
        if other is None:
            continue
        ox, oy, osize = other
        threshold = max(12.0, min(size, osize) * 0.55)
        if abs(cx - ox) <= threshold and abs(cy - oy) <= threshold:
            return
    faces.append(face)


def _expanded_face_crop_from_anime_head(head, image_width, image_height):
    bbox = head.get("bbox", []) if isinstance(head, dict) else []
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = w * 0.45, h * 0.45
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(int(image_width), int(round(x2 + px)))
    cy2 = min(int(image_height), int(round(y2 + py)))
    if cx2 - cx1 < 20 or cy2 - cy1 < 20:
        return None
    return cx1, cy1, cx2, cy2


def _face_detection_crop_scale(crop):
    if crop is None or crop.size == 0:
        return 1.0
    short = max(1, min(int(crop.shape[0]), int(crop.shape[1])))
    return 1.0 if short >= 256 else min(2.0, 256.0 / float(short))


def _rescale_face_detection(face, scale, offset_x=0.0, offset_y=0.0):
    if face is None:
        return None
    face = dict(face)
    bbox = list(face.get("bbox", []))
    if len(bbox) >= 4:
        for idx in range(4):
            bbox[idx] = float(bbox[idx]) / scale + (float(offset_x) if idx in (0, 2) else float(offset_y))
        face["bbox"] = bbox
    keypoints = []
    for point in face.get("keypoints", []) or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        new_point = list(point)
        new_point[0] = float(new_point[0]) / scale + float(offset_x)
        new_point[1] = float(new_point[1]) / scale + float(offset_y)
        keypoints.append(new_point)
    face["keypoints"] = keypoints
    return face


def _detect_faces_in_crop(detector, crop, offset_x=0.0, offset_y=0.0):
    if crop is None or crop.size == 0:
        return []
    import cv2
    scale = _face_detection_crop_scale(crop)
    detector_input = crop
    if scale > 1.01:
        detector_input = cv2.resize(crop, (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
    faces = []
    for pred in detector(detector_input) or []:
        face = _rescale_face_detection(_anime_face_from_prediction(pred), scale, offset_x, offset_y)
        if face is not None:
            face["source"] = "anime-face-detector"
            faces.append(face)
    return faces


def _anime_head_box_in_crop(head, crop_box, scale=1.0):
    bbox = head.get("bbox", []) if isinstance(head, dict) else []
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx1, cy1, cx2, cy2 = [float(v) for v in crop_box]
    x1 = max(cx1, min(cx2 - 1.0, x1)); y1 = max(cy1, min(cy2 - 1.0, y1))
    x2 = max(cx1 + 1.0, min(cx2, x2)); y2 = max(cy1 + 1.0, min(cy2, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    score = float(head.get("confidence", 1.0))
    return np.asarray([(x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale, score], dtype=np.float32)


def _detect_face_landmarks_for_anime_head(detector, crop, crop_box, head):
    if crop is None or crop.size == 0:
        return []
    import cv2
    scale = _face_detection_crop_scale(crop)
    detector_input = crop
    if scale > 1.01:
        detector_input = cv2.resize(crop, (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
    box = _anime_head_box_in_crop(head, crop_box, scale)
    if box is None:
        return []
    faces = []
    for pred in detector(detector_input, boxes=[box]) or []:
        face = _rescale_face_detection(_anime_face_from_prediction(pred), scale, crop_box[0], crop_box[1])
        if face is not None:
            face["source"] = "animeheads-guided"
            faces.append(face)
    return faces


def _filter_faces_by_heads(faces, heads):
    if not faces or not heads:
        return faces
    matched = {}
    for face in faces:
        fc = _face_bbox_center(face)
        if fc is None:
            continue
        fcx, fcy, fsize = fc
        best_i, best_score = None, None
        for i, head in enumerate(heads):
            x1, y1, x2, y2 = [float(v) for v in head.get("bbox", [])[:4]]
            hw, hh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            if fcx < x1 - hw * 0.35 or fcx > x2 + hw * 0.35 or fcy < y1 - hh * 0.35 or fcy > y2 + hh * 0.35:
                continue
            ratio = fsize / max(hw, hh)
            if ratio < 0.10 or ratio > 1.60:
                continue
            score = ((fcx - (x1 + x2) * 0.5) / max(hw, hh)) ** 2 + ((fcy - (y1 + y2) * 0.5) / max(hw, hh)) ** 2
            if best_score is None or score < best_score:
                best_i, best_score = i, score
        if best_i is not None:
            current = matched.get(best_i)
            if current is None or best_score < current[0]:
                matched[best_i] = (best_score, face)
    return [item[1] for _, item in sorted(matched.items(), key=lambda p: p[0])] or faces


def _detect_anime_face_keypoints(image_bgr, device="cuda", pose_keypoint=None, head_confidence=0.25):
    detector = _load_anime_face_detector(device)
    if detector is None or image_bgr is None:
        return []
    detections = []
    heads = []
    with torch.no_grad():
        for face in _detect_faces_in_crop(detector, image_bgr):
            _append_unique_face(detections, face)
        h, w = image_bgr.shape[:2]
        try:
            heads = _detect_anime_heads(image_bgr, ANIMEHEADS_MODEL_RELATIVE_PATH, max(0.20, float(head_confidence) * 0.8)) or []
        except Exception as e:
            print(f"[SAM3DBodyOpenPose] AnimeHeads-guided face landmark detection skipped. Error: {e}")
        for head in heads:
            crop_box = _expanded_face_crop_from_anime_head(head, w, h)
            if crop_box is None:
                continue
            x1, y1, x2, y2 = crop_box
            crop = image_bgr[y1:y2, x1:x2, :]
            for face in _detect_face_landmarks_for_anime_head(detector, crop, crop_box, head):
                _append_unique_face(detections, face)
    detections = _filter_faces_by_heads(detections, heads)
    detections.sort(key=lambda f: (float(f["bbox"][1]) if len(f.get("bbox", [])) >= 4 else 0.0, float(f["bbox"][0]) if len(f.get("bbox", [])) >= 4 else 0.0))
    return detections


def _point_from_body(person, index):
    body = person.get("pose_keypoints_2d", []) if isinstance(person, dict) else []
    off = index * 3
    if off + 2 >= len(body):
        return None
    try:
        x, y, c = float(body[off]), float(body[off + 1]), float(body[off + 2])
    except Exception:
        return None
    if c <= 0.0 or not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.asarray([x, y], dtype=np.float32)


def _pose_person_head_anchor(person):
    pts = [_point_from_body(person, idx) for idx in (0, 14, 15, 16, 17)]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    anchor = arr.mean(axis=0)
    head_size = 0.0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            head_size = max(head_size, float(np.linalg.norm(arr[i] - arr[j])))
    neck = _point_from_body(person, 1)
    neck_dist = float(np.linalg.norm(anchor - neck)) if neck is not None else 0.0
    radius = max(36.0, head_size * 2.8, neck_dist * 1.35)
    return float(anchor[0]), float(anchor[1]), radius


def _face_landmark_center(face):
    pts = []
    for p in face.get("keypoints", []) if isinstance(face, dict) else []:
        if len(p) < 2:
            continue
        score = float(p[2]) if len(p) > 2 else 1.0
        if score <= 0:
            continue
        x, y = float(p[0]), float(p[1])
        if np.isfinite(x) and np.isfinite(y):
            pts.append((x, y))
    if pts:
        arr = np.asarray(pts, dtype=np.float32)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    center = _face_bbox_center(face)
    return None if center is None else (center[0], center[1])


def _face_detection_confidence(face):
    bbox = face.get("bbox", []) if isinstance(face, dict) else []
    if len(bbox) >= 5:
        try:
            return float(bbox[4])
        except Exception:
            pass
    return 1.0


def _face_matches_pose_head(face, anchor):
    center = _face_landmark_center(face)
    box = _face_bbox_center(face)
    if center is None or box is None or anchor is None:
        return False
    if _face_detection_confidence(face) < 0.55:
        return False
    fcx, fcy = center
    ax, ay, radius = anchor
    fsize = float(box[2])
    return float(np.hypot(fcx - ax, fcy - ay)) <= min(radius + fsize * 0.20, radius * 1.05)


def _face_near_pose_head(face, anchor):
    center = _face_landmark_center(face)
    box = _face_bbox_center(face)
    if center is None or box is None or anchor is None:
        return False
    return float(np.hypot(center[0] - anchor[0], center[1] - anchor[1])) <= anchor[2] + float(box[2]) * 0.65


def _animehead_matches_pose_head(head, anchor):
    box = head.get("bbox", []) if isinstance(head, dict) else []
    if len(box) < 4 or anchor is None:
        return False
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    size = max(1.0, x2 - x1, y2 - y1)
    return float(np.hypot(cx - anchor[0], cy - anchor[1])) <= max(anchor[2], size * 0.85)


def _face_visibility_box(face):
    bbox = face.get("bbox", []) if isinstance(face, dict) else []
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return x1 - w * 0.18, y1 - h * 0.22, x2 + w * 0.18, y2 + h * 0.22


def _visible_face_for_pose_head(faces, anchor):
    best, best_d = None, None
    for face in faces or []:
        if isinstance(face, dict) and face.get("source") == "animeheads-guided":
            continue
        if not _face_matches_pose_head(face, anchor):
            continue
        center = _face_landmark_center(face)
        if center is None:
            continue
        dist = float(np.hypot(center[0] - anchor[0], center[1] - anchor[1]))
        if best_d is None or dist < best_d:
            best, best_d = face, dist
    return best


def _hide_invisible_openpose_face_parts(person, visible_face=None):
    body = list(person.get("pose_keypoints_2d", [])) if isinstance(person, dict) else []
    if not body:
        return person, False, 0
    box = _face_visibility_box(visible_face) if visible_face is not None else None
    changed, hidden = False, 0
    for idx in (0, 14, 15, 16, 17):
        off = idx * 3
        if off + 2 >= len(body):
            continue
        try:
            score = float(body[off + 2]); x = float(body[off]); y = float(body[off + 1])
        except Exception:
            continue
        if score <= 0:
            continue
        visible = False
        if box is not None:
            x1, y1, x2, y2 = box
            visible = x1 <= x <= x2 and y1 <= y <= y2
        if visible:
            continue
        body[off + 2] = 0.0
        changed = True
        hidden += 1
    if not changed:
        return person, False, 0
    new_person = dict(person)
    new_person["pose_keypoints_2d"] = body
    return new_person, True, hidden


def _remove_hidden_face_landmarks(pose_keypoint, source_image, batch_index, head_confidence=0.25):
    if not isinstance(pose_keypoint, dict):
        return pose_keypoint
    people = pose_keypoint.get("people", [])
    if not people:
        return pose_keypoint
    image_bgr = _comfy_image_to_bgr(source_image, batch_index)
    faces_meta = pose_keypoint.get("face_keypoints", {})
    faces = faces_meta.get("faces", []) if isinstance(faces_meta, dict) else []
    try:
        heads = _detect_anime_heads(image_bgr, ANIMEHEADS_MODEL_RELATIVE_PATH, head_confidence) or []
    except Exception as e:
        print(f"[SAM3DBodyOpenPose] hidden face part cleanup skipped. Error: {e}")
        return pose_keypoint
    new_people, hidden_anchors = [], []
    changed, hidden_parts = False, 0
    for person in people:
        anchor = _pose_person_head_anchor(person)
        if anchor is None or not any(_animehead_matches_pose_head(head, anchor) for head in heads):
            new_people.append(person)
            continue
        visible_face = _visible_face_for_pose_head(faces, anchor)
        new_person, did_hide, count = _hide_invisible_openpose_face_parts(person, visible_face)
        if did_hide:
            changed = True
            hidden_parts += count
        if visible_face is None:
            hidden_anchors.append(anchor)
        new_people.append(new_person)
    kept_faces, removed_faces = [], 0
    for face in faces:
        if any(_face_near_pose_head(face, anchor) for anchor in hidden_anchors):
            removed_faces += 1
            changed = True
        else:
            kept_faces.append(face)
    if not changed:
        return pose_keypoint
    print(f"[SAM3DBodyOpenPose] Hidden invisible OpenPose nose/eye/ear parts: {hidden_parts}; removed {removed_faces} hidden face landmarks.")
    out = dict(pose_keypoint)
    out["people"] = new_people
    if isinstance(faces_meta, dict):
        meta = dict(faces_meta)
        meta["faces"] = kept_faces
        if kept_faces:
            out["face_keypoints"] = meta
        else:
            out.pop("face_keypoints", None)
    return out


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
    key = (model_config["ckpt_path"], model_config.get("device", "cuda"))
    if key in _SAM3D_MODEL_CACHE:
        return _SAM3D_MODEL_CACHE[key]
    if LOCAL_SAM3D_DIR not in sys.path:
        sys.path.insert(0, LOCAL_SAM3D_DIR)
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


def _annotate_face_keypoints(pose_keypoint, source_image, batch_index, device="cuda", head_confidence=0.25):
    image_bgr = _comfy_image_to_bgr(source_image, batch_index)
    faces = _detect_anime_face_keypoints(image_bgr, device, pose_keypoint, head_confidence)
    if faces:
        pose_keypoint["face_keypoints"] = {"backend": "anime-face-detector", "model": "hysts/anime-face-detector", "faces": faces}
    return pose_keypoint


def _run_sam3d_estimator(estimator, image, bbox_threshold, inference_type, batch_index=0, head_confidence=0.25, device="cuda"):
    img_rgb = _comfy_image_to_rgb(image, batch_index)
    img_bgr = img_rgb[..., ::-1].copy()
    height, width = img_rgb.shape[:2]
    bboxes = None
    try:
        bboxes, _targets = _body_boxes_from_heads_and_rtdetr(img_bgr, head_confidence, device)
        if bboxes is not None and len(bboxes) > 0:
            print(f"[SAM3DBodyOpenPose] Using AnimeHeads/RT-DETRv4 guided person boxes: {len(bboxes)}")
    except Exception as e:
        print(f"[SAM3DBodyOpenPose] guided person boxes skipped; using full image. Error: {e}")
        bboxes = None
    outputs = estimator.process_one_image(img_rgb, bboxes=bboxes, bbox_thr=bbox_threshold, inference_type=inference_type)
    if not outputs:
        outputs = estimator.process_one_image(img_rgb, bboxes=None, bbox_thr=bbox_threshold, inference_type=inference_type)
    if not outputs:
        raise RuntimeError("No people detected in image")
    return _sort_outputs_top_left(outputs), width, height


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
                "head_confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"body_shape_image": ("IMAGE",)},
        }
    RETURN_TYPES = ("POSE_KEYPOINT", "POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES = ("pose_keypoint", "body_shape_keypoint", "debug_person_crops")
    FUNCTION = "analyze"
    CATEGORY = "SAM3DBody/openpose"

    def analyze(self, pose_image, device="cuda", bbox_threshold=0.8, inference_type="full", confidence=1.0, auto_head_proportion=True, head_confidence=0.25, body_shape_image=None):
        model_config = _make_model_config(DEFAULT_MODEL_PATH, device)
        loaded = _load_sam3d_model(model_config)
        from sam_3d_body import SAM3DBodyEstimator
        estimator = SAM3DBodyEstimator(loaded["model"], loaded["model_cfg"], human_detector=None, human_segmentor=None, fov_estimator=None)
        pose_outputs, pose_width, pose_height = _run_sam3d_estimator(estimator, pose_image, bbox_threshold, inference_type, 0, head_confidence, device)
        pose_pose = _make_openpose_keypoint_from_outputs(pose_outputs, pose_width, pose_height, confidence)
        sam31_used = False
        try:
            from .sam31_face import detect_parts, update_pose
            anchors = [_pose_person_head_anchor(p) for p in pose_pose.get("people", [])]
            if anchors:
                rgb = _comfy_image_to_rgb(pose_image, 0)
                parts = detect_parts(rgb, anchors)
                if any(candidates for head in parts for candidates in head.values()):
                    pose_pose = update_pose(pose_pose, parts)
                    sam31_used = True
                else:
                    print("[SAM3DBodyOpenPose] SAM3.1 found no face parts; using anime-face-detector fallback.")
        except Exception as e:
            print(f"[SAM3DBodyOpenPose] ComfyUI SAM3.1 face-part detection unavailable; using anime-face-detector. Error: {e}")
        if not sam31_used:
            try:
                pose_pose = _annotate_face_keypoints(pose_pose, pose_image, 0, device, head_confidence)
                pose_pose = _remove_hidden_face_landmarks(pose_pose, pose_image, 0, head_confidence)
            except Exception as e:
                print(f"[SAM3DBodyOpenPose] face landmark fallback skipped. Error: {e}")
        body_shape_outputs = pose_outputs
        body_shape_face_people = []
        if body_shape_image is not None:
            collected = []
            for batch_index in range(_image_batch_size(body_shape_image)):
                try:
                    outs, shape_width, shape_height = _run_sam3d_estimator(estimator, body_shape_image, bbox_threshold, inference_type, batch_index, head_confidence, device)
                    collected.extend(outs)
                    if auto_head_proportion:
                        shape_pose = _make_openpose_keypoint_from_outputs(outs, shape_width, shape_height, confidence)
                        try:
                            from .sam31_face import detect_parts, update_pose
                            shape_anchors = [_pose_person_head_anchor(p) for p in shape_pose["people"]]
                            shape_parts = detect_parts(_comfy_image_to_rgb(body_shape_image, batch_index), shape_anchors)
                            shape_pose = update_pose(shape_pose, shape_parts)
                        except Exception as e:
                            print(f"[SAM3DBodyOpenPose] body-shape face detection skipped for batch {batch_index}. Error: {e}")
                        body_shape_face_people.extend(shape_pose["people"])
                except Exception as e:
                    print(f"[SAM3DBodyOpenPose] body_shape_image batch {batch_index} skipped. Error: {e}")
            if collected:
                body_shape_outputs = collected[:len(pose_outputs)]
        body_shape_pose = _make_body_shape_keypoint_from_outputs(body_shape_outputs, pose_width, pose_height, confidence)
        if auto_head_proportion:
            from .sam31_face import adapt_pose_face_to_body_shape, match_body_shape_head_to_pose
            if body_shape_face_people:
                pose_pose = adapt_pose_face_to_body_shape(pose_pose, body_shape_face_people[:len(pose_outputs)])
            body_shape_pose = match_body_shape_head_to_pose(body_shape_pose, pose_pose)
        return ([pose_pose], [body_shape_pose], _debug_crops_to_image_tensor([]))


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
        pose_result = util.draw_pose_json(json.dumps(draw_frames), render_width, use_ground_plane, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, pelvis_scale, torso_scale, neck_scale, head_scale, eye_distance_scale, 0.0, 0.0, left_eye_scale, right_eye_scale, left_eyebrow_scale, right_eyebrow_scale, mouth_scale, nose_scale_face, face_shape_scale, shoulder_scale, arm_scale, leg_scale, hands_scale, overall_scale, 0.0, offset_x, offset_y, target_pose_keypoint_obj=body_shape_keypoint)
        pose_imgs = pose_result[0] or [np.zeros((render_height, render_width, 3), dtype=np.uint8)]
        if render_height > 0:
            import cv2
            pose_imgs = [cv2.resize(img, (render_width, render_height), interpolation=cv2.INTER_AREA) if img.shape[0] != render_height or img.shape[1] != render_width else img for img in pose_imgs]
        return (torch.from_numpy(np.asarray(pose_imgs).astype(np.float32) / 255.0),)


NODE_CLASS_MAPPINGS = {"SAM3DBodyOpenPoseAnalyze": SAM3DBodyOpenPoseAnalyze, "SAM3DBodyOpenPoseRender": SAM3DBodyOpenPoseRender}
NODE_DISPLAY_NAME_MAPPINGS = {"SAM3DBodyOpenPoseAnalyze": "SAM3D Body OpenPose Analyze", "SAM3DBodyOpenPoseRender": "SAM3D Body OpenPose Render"}
