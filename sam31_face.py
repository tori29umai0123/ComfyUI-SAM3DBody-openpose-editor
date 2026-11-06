"""Face-part detection using ComfyUI's native SAM3.1 model and text encoder."""

import os
import math
import shutil
import tempfile
from threading import Lock

import numpy as np
import torch


PART_PROMPTS = {
    "person": "person",
    "face": "face",
    "eye": "eye",
    "closed_eye": "closed eye",
    "eyebrow": "eyebrow",
    "nose": "nose",
    "mouth": "mouth",
    "closed_mouth": "closed mouth",
    "ear": "ear",
    "eyepatch": "eye patch",
    "glasses": "glasses",
    "mask": "face mask",
}
_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "sam3.1", "sam3.1_multiplex_fp16.safetensors")
_CHECKPOINT_REPO = "Comfy-Org/sam3.1"
_CHECKPOINT_FILENAME = "checkpoints/sam3.1_multiplex_fp16.safetensors"
_download_lock = Lock()
_loaded = None


def ensure_checkpoint():
    with _download_lock:
        if os.path.isfile(_CHECKPOINT) and os.path.getsize(_CHECKPOINT) > 0:
            return _CHECKPOINT
        try:
            if __package__:
                from .model_download import download
            else:  # Direct import by the standalone test suite.
                from model_download import download

            os.makedirs(os.path.dirname(_CHECKPOINT), exist_ok=True)
            print(f"[SAM3DBodyOpenPose] Downloading SAM3.1 checkpoint to {_CHECKPOINT}")
            downloaded = download("file", repo_id=_CHECKPOINT_REPO, filename=_CHECKPOINT_FILENAME)
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(_CHECKPOINT), suffix=".partial", delete=False) as temporary:
                temporary_path = temporary.name
            try:
                shutil.copyfile(downloaded, temporary_path)
                os.replace(temporary_path, _CHECKPOINT)
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        except Exception as e:
            raise RuntimeError(f"SAM3.1 checkpoint download failed. Place it at {_CHECKPOINT}. Error: {e}") from e
        return _CHECKPOINT


def load_model_and_clip():
    global _loaded
    if _loaded is None:
        import comfy.sd
        model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(ensure_checkpoint(), output_vae=False)
        if model is None or clip is None:
            raise RuntimeError("SAM3.1 checkpoint did not provide both model and text encoder")
        _loaded = model, clip
    return _loaded


def _text_embedding(clip, prompt, device, dtype):
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt), show_pbar=False)
    emb, meta = conditioning[0]
    emb = emb.to(device=device, dtype=dtype)
    mask = meta.get("attention_mask")
    if mask is None:
        mask = torch.ones(emb.shape[:2], dtype=torch.int64, device=device)
    else:
        mask = mask.to(device)
    return emb, mask


def _crop_from_anchor(anchor, width, height):
    x, y, radius = anchor
    radius = max(24.0, min(radius * .9, max(width, height) * 0.3))
    x1, y1 = max(0, int(x - radius)), max(0, int(y - radius))
    x2, y2 = min(width, int(x + radius)), min(height, int(y + radius))
    return (x1, y1, x2, y2) if x2 - x1 >= 24 and y2 - y1 >= 24 else None


def _part_candidates(results, crop, threshold, max_fraction=0.55, include_outline=False):
    x1, y1, x2, y2 = crop
    scores = results["scores"][0].sigmoid().detach().float().cpu().numpy()
    boxes = results["boxes"][0].detach().float().cpu().numpy()
    found = []
    masks = results.get("masks") if include_outline else None
    for candidate_index, (box, score) in enumerate(zip(boxes, scores)):
        if score < threshold:
            continue
        bx1, by1, bx2, by2 = [float(v) for v in box]
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 1 or bh <= 1 or bw > (x2-x1) * max_fraction or bh > (y2-y1) * max_fraction:
            continue
        cx, cy = (bx1 + bx2) / 2 + x1, (by1 + by2) / 2 + y1
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        item = {"x": cx, "y": cy, "score": float(score), "box": [bx1+x1, by1+y1, bx2+x1, by2+y1]}
        if masks is not None:
            import cv2
            mask = masks[0][candidate_index].detach().float().cpu().numpy().squeeze()
            if mask.ndim == 2:
                mask = cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(contour) > 20:
                        contour = cv2.approxPolyDP(contour, max(1.0, cv2.arcLength(contour, True) * .003), True)[:, 0, :]
                        if len(contour) > 256:
                            contour = contour[np.linspace(0, len(contour)-1, 256).astype(int)]
                        item["outline"] = [[float(px+x1), float(py+y1)] for px, py in contour]
        found.append(item)
    found.sort(key=lambda p: p["score"], reverse=True)
    kept = []
    for p in found:
        if all(np.hypot(p["x"]-q["x"], p["y"]-q["y"]) > max(3, min(x2-x1, y2-y1) * .07) for q in kept):
            kept.append(p)
    return kept


def _person_candidates(results, width, height, threshold):
    """Extract SAM3.1 person boxes and masks in full-image coordinates."""
    import cv2

    scores = results["scores"][0].sigmoid().detach().float().cpu().numpy()
    boxes = results["boxes"][0].detach().float().cpu().numpy()
    masks = results.get("masks")
    if masks is None:
        return []
    found = []
    for index, (box, score) in enumerate(zip(boxes, scores)):
        if score < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 - x1 < 12 or y2 - y1 < 12:
            continue
        mask = masks[0][index].detach().float().cpu().numpy().squeeze()
        if mask.ndim != 2:
            continue
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = mask > (0.5 if mask.min() >= 0 and mask.max() <= 1 else 0.0)
        if np.count_nonzero(mask) < 100:
            continue
        found.append({"box": [x1, y1, x2, y2], "mask": mask, "score": float(score)})
    return sorted(found, key=lambda item: item["score"], reverse=True)


def _box_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(1.0, (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - intersection)
    return intersection / union


def _assign_person_regions(anchors, person_boxes, candidates):
    """Match SAM3.1 person masks to SAM3D people without reusing a mask."""
    pairs = []
    for person_index, anchor in enumerate(anchors):
        if anchor is None:
            continue
        x, y, radius = anchor
        reference = person_boxes[person_index] if person_boxes and person_index < len(person_boxes) else None
        for candidate_index, candidate in enumerate(candidates):
            box = candidate["box"]
            height, width = candidate["mask"].shape
            ix, iy = int(round(x)), int(round(y))
            on_mask = 0 <= ix < width and 0 <= iy < height and bool(candidate["mask"][iy, ix])
            in_box = box[0] <= x <= box[2] and box[1] <= y <= box[3]
            overlap = _box_iou(reference, box)
            if not (on_mask or in_box or overlap > 0.05):
                continue
            distance = np.hypot(x - (box[0]+box[2])*0.5, y - box[1]) / max(radius, box[2]-box[0], 1.0)
            score = 4.0 * overlap + 2.0 * on_mask + 0.5 * in_box - 0.1 * distance
            pairs.append((score, person_index, candidate_index))
    assigned = [None] * len(anchors)
    used = set()
    for _, person_index, candidate_index in sorted(pairs, reverse=True):
        if assigned[person_index] is None and candidate_index not in used:
            assigned[person_index] = candidates[candidate_index]
            used.add(candidate_index)
    return assigned


def _masked_person_crop(image, crop, region):
    """Hide neighboring people before running face-part prompts."""
    import cv2

    x1, y1, x2, y2 = crop
    frame = image[y1:y2, x1:x2, :3].copy()
    if region is None:
        return frame
    mask = region["mask"][y1:y2, x1:x2].astype(np.uint8)
    margin = max(3, int(round(min(x2-x1, y2-y1) * 0.06)))
    mask = cv2.dilate(mask, np.ones((margin, margin), dtype=np.uint8)) > 0
    frame[~mask] = 114
    return frame


def detect_people(image, threshold=0.5):
    """Detect full-image person instances for optional mask-conditioned SAM3DBody inference."""
    import comfy.model_management
    import comfy.utils

    model, clip = load_model_and_clip()
    if getattr(model.model.model_config, "unet_config", {}).get("image_model") != "SAM31":
        raise ValueError("The person detector requires a SAM3.1 MODEL")
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    h, w = image.shape[:2]
    frame = torch.from_numpy(image[:, :, :3].copy()).float().div_(255).permute(2, 0, 1).unsqueeze(0)
    frame = comfy.utils.common_upscale(frame, 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)
    embedding, text_mask = _text_embedding(clip, PART_PROMPTS["person"], device, dtype)
    with torch.inference_mode():
        result = model.model.diffusion_model(frame, text_embeddings=embedding, text_mask=text_mask,
                                             threshold=threshold, orig_size=(h, w))
    return _person_candidates(result, w, h, threshold)


def detect_parts(image, anchors, threshold=0.5, person_boxes=None, person_candidates=None):
    """Return per-head detections in original image coordinates."""
    import comfy.model_management
    import comfy.utils
    model, clip = load_model_and_clip()
    if getattr(model.model.model_config, "unet_config", {}).get("image_model") != "SAM31":
        raise ValueError("The face detector requires a SAM3.1 MODEL")
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    network = model.model.diffusion_model
    h, w = image.shape[:2]
    embeddings = {name: _text_embedding(clip, prompt, device, dtype)
                  for name, prompt in PART_PROMPTS.items() if name != "person" or len(anchors) > 1}
    detections = []
    with torch.inference_mode():
        regions = [None] * len(anchors)
        if len(anchors) > 1:
            try:
                if person_candidates is None:
                    full_frame = torch.from_numpy(image[:, :, :3].copy()).float().div_(255).permute(2, 0, 1).unsqueeze(0)
                    full_frame = comfy.utils.common_upscale(full_frame, 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)
                    emb, text_mask = embeddings["person"]
                    person_result = network(full_frame, text_embeddings=emb, text_mask=text_mask,
                                            threshold=threshold, orig_size=(h, w))
                    candidates = _person_candidates(person_result, w, h, threshold)
                else:
                    candidates = person_candidates
                regions = _assign_person_regions(anchors, person_boxes, candidates)
                print(f"[SAM3DBodyOpenPose] SAM3.1 person masks matched: {sum(region is not None for region in regions)}/{len(anchors)}")
            except Exception as error:
                print(f"[SAM3DBodyOpenPose] SAM3.1 person masks unavailable; using head crops. Error: {error}")
        for person_index, anchor in enumerate(anchors):
            crop = _crop_from_anchor(anchor, w, h) if anchor is not None else None
            if crop is None:
                detections.append({})
                continue
            x1, y1, x2, y2 = crop
            person_image = _masked_person_crop(image, crop, regions[person_index])
            frame = torch.from_numpy(person_image).float().div_(255).permute(2, 0, 1).unsqueeze(0)
            frame = comfy.utils.common_upscale(frame, 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)
            parts = {}
            for name in ("face", "eye", "closed_eye", "eyebrow", "nose", "mouth", "closed_mouth", "ear"):
                emb, mask = embeddings[name]
                result = network(frame, text_embeddings=emb, text_mask=mask, threshold=threshold, orig_size=(y2-y1, x2-x1))
                parts[name] = _part_candidates(result, crop, threshold, 0.95 if name == "face" else 0.55,
                                               name in ("face", "eye", "closed_eye", "eyebrow", "mouth", "closed_mouth"))
            needed = (["eyepatch", "glasses"] if len(parts["eye"]) + len(parts["closed_eye"]) < 2 else [])
            if not parts["nose"] or not parts["mouth"]:
                needed.append("mask")
            for name in needed:
                emb, mask = embeddings[name]
                result = network(frame, text_embeddings=emb, text_mask=mask, threshold=threshold, orig_size=(y2-y1, x2-x1))
                parts[name] = _part_candidates(result, crop, threshold)
            detections.append(parts)
    return detections


def _occluded(point, occluders):
    return any(b[0] <= point["x"] <= b[2] and b[1] <= point["y"] <= b[3]
               for item in occluders for b in [item["box"]])


def _is_closed_feature(item, limit):
    outline = np.asarray(item.get("outline", []), dtype=np.float32)
    if outline.ndim == 2 and outline.shape[0] >= 5 and outline.shape[1] == 2:
        import cv2
        _, (width, height), _ = cv2.minAreaRect(outline)
        if max(width, height) > 0:
            return min(width, height) / max(width, height) < limit
    x1, y1, x2, y2 = item["box"]
    return (y2-y1) / max(1.0, x2-x1) < limit


def _merge_eyes(open_eyes, closed_eyes):
    candidates = [{**item, "closed": _is_closed_feature(item, .38)} for item in open_eyes + closed_eyes]
    kept = []
    for item in sorted(candidates, key=lambda p: p["score"], reverse=True):
        overlapping = next((index for index, other in enumerate(kept)
                            if np.hypot(item["x"]-other["x"], item["y"]-other["y"])
                            < max(4.0, (item["box"][2]-item["box"][0]) * .55)), None)
        if overlapping is None:
            kept.append(item)
        elif item["closed"] and not kept[overlapping]["closed"] and item["score"] >= kept[overlapping]["score"] * .75:
            kept[overlapping] = item
    return sorted(kept, key=lambda p: p["score"], reverse=True)[:2]


def _merge_mouth(open_mouth, closed_mouth):
    candidates = [{**item, "closed": _is_closed_feature(item, .26)} for item in open_mouth + closed_mouth]
    if not candidates:
        return []
    best = max(candidates, key=lambda p: p["score"])
    if not best["closed"]:
        narrow = [item for item in candidates if item["closed"] and item["score"] >= best["score"] * .75
                  and np.hypot(item["x"]-best["x"], item["y"]-best["y"]) < max(5.0, best["box"][2]-best["box"][0]) * .5]
        if narrow:
            best = max(narrow, key=lambda p: p["score"])
    return [best]


def _visible_ear_candidates(parts):
    """Reject ear text detections that lie in the forehead or middle of the face."""
    ears = parts.get("ear", [])
    eyes = sorted(parts.get("eye", []), key=lambda item: item["x"])
    if len(eyes) < 2:
        if eyes and parts.get("nose"):
            eye, nose = eyes[0], parts["nose"][0]
            distance = max(1.0, float(np.hypot(eye["x"]-nose["x"], eye["y"]-nose["y"])))
            return [ear for ear in ears
                    if np.hypot(ear["x"]-nose["x"], ear["y"]-nose["y"]) > distance * 1.2
                    and abs(ear["y"]-eye["y"]) < distance * 1.2]
        return [ear for ear in ears if ear["score"] >= .85]
    left, right = eyes[:2]
    center = np.array([(left["x"]+right["x"])/2, (left["y"]+right["y"])/2])
    horizontal = np.array([right["x"]-left["x"], right["y"]-left["y"]], dtype=float)
    span = float(np.linalg.norm(horizontal))
    if span < 2:
        return []
    horizontal /= span
    vertical = np.array([-horizontal[1], horizontal[0]])
    mouth = (parts.get("mouth") or [None])[0]
    if mouth and np.dot(np.array([mouth["x"], mouth["y"]])-center, vertical) < 0:
        vertical = -vertical
    visible = []
    for ear in ears:
        relative = np.array([ear["x"], ear["y"]]) - center
        lateral, down = float(np.dot(relative, horizontal)), float(np.dot(relative, vertical))
        if abs(lateral) >= span * .58 and -span * .35 <= down <= span * .9:
            visible.append(ear)
    return visible


def _select_or_infer_brows(parts):
    """Keep eyebrows separate from lashes; estimate them above visible eyes if needed."""
    eyes = sorted(parts.get("eye", []), key=lambda item: item["x"])
    if not eyes:
        return []
    if len(eyes) >= 2:
        axis = np.array([eyes[1]["x"]-eyes[0]["x"], eyes[1]["y"]-eyes[0]["y"]], dtype=float)
        span = float(np.linalg.norm(axis))
        if span < 3:
            return []
        axis /= span
    else:
        axis = np.array([1.0, 0.0])
        box = eyes[0]["box"]
        span = max(10.0, (box[2]-box[0]) * 2.5)
    down = np.array([-axis[1], axis[0]])
    eye_center = np.mean([[eye["x"], eye["y"]] for eye in eyes], axis=0)
    lower = (parts.get("mouth") or parts.get("nose") or [None])[0]
    if lower is not None and np.dot(np.array([lower["x"], lower["y"]])-eye_center, down) < 0:
        down = -down
    selected = []
    remaining = list(parts.get("eyebrow", []))
    for eye in eyes[:2]:
        eye_xy = np.array([eye["x"], eye["y"]], dtype=float)
        box = eye["box"]
        eye_width = max(1.0, float(box[2]-box[0]))
        eligible = []
        for brow in remaining:
            relative = np.array([brow["x"], brow["y"]]) - eye_xy
            above = -float(np.dot(relative, down))
            sideways = abs(float(np.dot(relative, axis)))
            width = brow["box"][2] - brow["box"][0]
            if (.32 * span <= above <= .85 * span and sideways <= .45 * span
                    and width <= max(eye_width * 1.55, span * .5)):
                eligible.append(brow)
        if eligible:
            brow = max(eligible, key=lambda item: item["score"])
            remaining.remove(brow)
            selected.append(brow)
            continue
        width = max((box[2]-box[0]) * 1.25, span * .31)
        center = eye_xy - down * span * .43 - axis * span * .08
        landmarks = [(center + axis * (width * t) + down * (width * curve)).tolist()
                     for t, curve in ((-.5, .08), (-.25, -.02), (0, -.07), (.25, -.02), (.5, .08))]
        coordinates = np.asarray(landmarks)
        selected.append({"x": float(center[0]), "y": float(center[1]), "score": .35,
                         "box": [*coordinates.min(axis=0).tolist(), *coordinates.max(axis=0).tolist()],
                         "landmarks": landmarks, "inferred": True})
    return selected


def _infer_missing_ears(body, parts):
    """Place hidden OpenPose ear anchors from visible eye and lower-face geometry."""
    visible = [index for index in (14, 15) if body[index * 3 + 2] > 0]
    if not visible:
        return body
    reference = (parts.get("mouth") or parts.get("nose") or [None])[0]
    if len(visible) == 2:
        left = np.asarray(body[14 * 3:14 * 3 + 2], dtype=float)
        right = np.asarray(body[15 * 3:15 * 3 + 2], dtype=float)
        direction = right - left
        span = float(np.linalg.norm(direction))
        if span < 3:
            return body
        direction /= span
        down = np.array([-direction[1], direction[0]])
        center = (left + right) / 2
        if reference is not None and np.dot(np.array([reference["x"], reference["y"]])-center, down) < 0:
            down = -down
        side_offsets = {-1: -span * .82, 1: span * .82}
        face = (parts.get("face") or [None])[0]
        outline = np.asarray(face.get("outline", []), dtype=float) if face else np.empty((0, 2))
        if outline.ndim == 2 and outline.shape[0] >= 12 and outline.shape[1] == 2:
            relative = outline - center
            local_x, local_y = relative @ direction, relative @ down
            near_ears = np.abs(local_y - span * .18) <= span * .48
            if near_ears.sum() >= 4:
                for sign in (-1, 1):
                    edge = float(np.min(local_x[near_ears]) if sign < 0 else np.max(local_x[near_ears]))
                    if edge * sign > span * .55:
                        side_offsets[sign] = sign * float(np.clip(edge * sign + span * .08,
                                                                  span * .72, span * 1.20))
        for index, sign in ((16, -1), (17, 1)):
            if body[index * 3 + 2] <= 0:
                estimated = center + direction * side_offsets[sign] + down * span * .18
                body[index * 3:index * 3 + 3] = [float(estimated[0]), float(estimated[1]), .25]
    elif reference is not None:
        eye_index = visible[0]
        ear_index = 16 if eye_index == 14 else 17
        if body[ear_index * 3 + 2] <= 0:
            eye = np.asarray(body[eye_index * 3:eye_index * 3 + 2], dtype=float)
            lower = np.array([reference["x"], reference["y"]], dtype=float)
            outward = eye - lower
            span = float(np.linalg.norm(outward))
            if span >= 3:
                down = np.array([0.0, 1.0])
                estimated = eye + outward * .7 + down * span * .35
                body[ear_index * 3:ear_index * 3 + 3] = [float(estimated[0]), float(estimated[1]), .2]
    return body


def _infer_missing_nose(body, parts):
    """Keep the central head bones when SAM labels the mouth as the nose."""
    if body[2] > 0 or any(body[index * 3 + 2] <= 0 for index in (14, 15)) or not parts.get("mouth"):
        return body
    eyes = (np.asarray(body[14 * 3:14 * 3 + 2], dtype=float)
            + np.asarray(body[15 * 3:15 * 3 + 2], dtype=float)) / 2
    mouth = parts["mouth"][0]
    lower = np.array([mouth["x"], mouth["y"]], dtype=float)
    if np.linalg.norm(lower - eyes) > 3:
        estimated = eyes + (lower - eyes) * .43
        body[:3] = [float(estimated[0]), float(estimated[1]), .25]
    return body


def _jaw_from_outline(face, left_eye, right_eye, mouth=None, nose=None):
    outline = np.asarray(face.get("outline", []), dtype=float)
    if outline.ndim != 2 or outline.shape[0] < 12 or outline.shape[1] != 2:
        return None
    origin = np.array([(left_eye["x"] + right_eye["x"]) / 2,
                       (left_eye["y"] + right_eye["y"]) / 2], dtype=float)
    horizontal = np.array([right_eye["x"]-left_eye["x"], right_eye["y"]-left_eye["y"]], dtype=float)
    length = float(np.linalg.norm(horizontal))
    if length < 3:
        return None
    horizontal /= length
    vertical = np.array([-horizontal[1], horizontal[0]])
    toward = mouth or nose
    if toward is not None and np.dot(np.array([toward["x"], toward["y"]]) - origin, vertical) < 0:
        vertical = -vertical
    relative = outline - origin
    local_x = relative @ horizontal
    local_y = relative @ vertical
    lower = local_y > max(0.0, float(np.percentile(local_y, 35)))
    if lower.sum() < 8:
        return None
    xs, ys = local_x[lower], local_y[lower]
    left, right = np.percentile(xs, [3, 97])
    if right-left < 8:
        return None
    band = max(2.0, (right-left) / 25)
    sampled_x, sampled_y = [], []
    for index in range(17):
        x = left + (right-left) * (.5-.5*math.cos(math.pi*index/16))
        near = np.abs(xs-x) <= band
        if not near.any():
            near = np.abs(xs-x) == np.min(np.abs(xs-x))
        sampled_x.append(x)
        sampled_y.append(float(np.max(ys[near])))
    for _ in range(2):
        sampled_y = [sampled_y[0]] + [sampled_y[i-1]*.2+sampled_y[i]*.6+sampled_y[i+1]*.2
                                      for i in range(1, 16)] + [sampled_y[-1]]
    return [(origin + horizontal*x + vertical*y).tolist() for x, y in zip(sampled_x, sampled_y)]


def _feature_outline_points(item, shape, horizontal, closed=False, interior=False):
    """Sample visible mask edges in the tilted face's coordinate frame."""
    outline = np.asarray(item.get("outline", []), dtype=float)
    if outline.ndim != 2 or outline.shape[0] < 5 or outline.shape[1] != 2:
        return None
    axis = np.asarray(horizontal, dtype=float)
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return None
    axis /= length
    down = np.array([-axis[1], axis[0]])
    center = outline.mean(axis=0)
    local = np.stack(((outline-center) @ axis, (outline-center) @ down), axis=1)
    low = local.min(axis=0)
    high = local.max(axis=0)
    if np.min(high-low) < 1:
        return None
    sampled = []
    for x, y in shape:
        target_x = low[0] + (high[0]-low[0])*x
        if interior:
            target_y = low[1] + (high[1]-low[1])*y
            result = center + axis*target_x + down*target_y
        elif closed:
            near = np.abs(local[:, 0]-target_x) <= max(1.5, (high[0]-low[0])*.08)
            if not near.any():
                near = np.abs(local[:, 0]-target_x) == np.min(np.abs(local[:, 0]-target_x))
            result = center + axis*target_x + down*float(np.mean(local[near, 1]))
        else:
            target_y = low[1] + (high[1]-low[1])*y
            # The selected point lies on the actual SAM mask boundary.
            index = int(np.argmin((local[:, 0]-target_x)**2 + (local[:, 1]-target_y)**2))
            result = outline[index]
        sampled.append(result.tolist())
    return sampled


def _pseudo_face_keypoints(parts, body):
    """Place the 70 OpenPose slots from visible SAM3.1 part boxes."""
    points = [[0.0, 0.0, 0.0] for _ in range(70)]

    def put(index, item, x, y):
        box = item.get("box", [])
        if len(box) != 4 or not np.isfinite(box).all():
            return
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            return
        points[index] = [x1 + (x2 - x1) * x, y1 + (y2 - y1) * y,
                         max(0.0, min(1.0, float(item["score"]) * 0.65))]

    def put_outline(index, item, x, y, sampled, offset):
        if sampled is None:
            put(index, item, x, y)
        else:
            sx, sy = sampled[offset]
            points[index] = [sx, sy, max(0.0, min(1.0, float(item["score"]) * .65))]

    eyes = sorted(sorted(parts.get("eye", []), key=lambda p: p["score"], reverse=True)[:2], key=lambda p: p["x"])
    eye_slots = [None, None]
    if len(eyes) == 2:
        eye_slots = eyes
    elif eyes:
        eye = eyes[0]
        distances = [(eye["x"] - body[i * 3]) ** 2 + (eye["y"] - body[i * 3 + 1]) ** 2 for i in (14, 15)]
        eye_slots[0 if distances[0] <= distances[1] else 1] = eye
    if all(eye_slots):
        horizontal = np.array([eye_slots[1]["x"]-eye_slots[0]["x"],
                               eye_slots[1]["y"]-eye_slots[0]["y"]], dtype=float)
    else:
        horizontal = np.array([1.0, 0.0])
    eye_shape = [(0.0, .5), (.25, .15), (.75, .15), (1.0, .5), (.75, .85), (.25, .85)]
    for slot, eye in enumerate(eye_slots):
        if eye is None:
            continue
        shape = ([(0, .52), (.25, .38), (.75, .38), (1, .52), (.75, .55), (.25, .55)]
                 if eye.get("closed") else eye_shape)
        sampled = _feature_outline_points(eye, shape, horizontal, bool(eye.get("closed")))
        for offset, (x, y) in enumerate(shape):
            put_outline((36, 42)[slot] + offset, eye, x, y, sampled, offset)
        if not eye.get("closed"):
            put((68, 69)[slot], eye, .5, .5)

    brows = sorted(sorted(parts.get("eyebrow", []), key=lambda p: p["score"], reverse=True)[:2], key=lambda p: p["x"])
    if len(brows) == 2:
        brow_slots = brows
    elif brows:
        brow_slots = [None, None]
        if eye_slots[0] is not None or eye_slots[1] is not None:
            distances = [abs(brows[0]["x"] - eye["x"]) if eye is not None else float("inf") for eye in eye_slots]
            brow_slots[0 if distances[0] <= distances[1] else 1] = brows[0]
        else:
            nose_x = (parts.get("nose") or [{}])[0].get("x", brows[0]["x"])
            brow_slots[0 if brows[0]["x"] < nose_x else 1] = brows[0]
    else:
        brow_slots = [None, None]
    for slot, brow in enumerate(brow_slots):
        if brow is None:
            continue
        brow_shape = ((0, .75), (.25, .4), (.5, .2), (.75, .4), (1, .75))
        sampled = brow.get("landmarks") or _feature_outline_points(brow, brow_shape, horizontal, closed=True)
        for offset, (x, y) in enumerate(brow_shape):
            put_outline((17, 22)[slot] + offset, brow, x, y, sampled, offset)

    if parts.get("nose"):
        nose = parts["nose"][0]
        nose_shape = [(.5, 0), (.5, .2), (.5, .4), (.5, .6),
                      (.1, .75), (.25, .9), (.5, 1), (.75, .9), (.9, .75)]
        for index, (x, y) in enumerate(nose_shape, 27):
            put(index, nose, x, y)

    if parts.get("mouth"):
        mouth = parts["mouth"][0]
        outer = [(0, .5), (.15, .3), (.3, .15), (.5, .1), (.7, .15), (.85, .3),
                 (1, .5), (.85, .7), (.7, .85), (.5, .9), (.3, .85), (.15, .7)]
        inner = [(.2, .5), (.35, .4), (.5, .38), (.65, .4),
                 (.8, .5), (.65, .62), (.5, .65), (.35, .62)]
        if mouth.get("closed"):
            outer = [(x, .5 - .12 * math.sin(math.pi*x)) for x, _ in outer]
            inner = []
        sampled = _feature_outline_points(mouth, outer, horizontal, bool(mouth.get("closed")))
        inner_sampled = _feature_outline_points(mouth, inner, horizontal, interior=True) if inner else None
        for index, (x, y) in enumerate(outer + inner, 48):
            if index - 48 < len(outer):
                put_outline(index, mouth, x, y, sampled, index - 48)
            else:
                put_outline(index, mouth, x, y, inner_sampled, index - 48 - len(outer))

    # A jaw outline is only sketched when both eyes and a face box are observed.
    if all(eye_slots) and parts.get("face"):
        face = parts["face"][0]
        jaw = _jaw_from_outline(face, eye_slots[0], eye_slots[1],
                                (parts.get("mouth") or [None])[0], (parts.get("nose") or [None])[0])
        for index in range(17):
            if jaw:
                x, y = jaw[index]
                points[index] = [x, y, max(0.0, min(1.0, face["score"] * .65))]
            else:
                t = index / 16.0
                angle = math.pi * t
                put(index, face, .5 - .5 * math.cos(angle), .48 + .50 * math.sin(angle))

    return [value for point in points for value in point]


def update_pose(pose, detections):
    """Replace face bone anchors with measured parts; infer only a hidden paired eye/ear."""
    out = dict(pose)
    people = []
    faces = []
    for person, parts in zip(pose.get("people", []), detections):
        person = dict(person)
        body = list(person.get("pose_keypoints_2d", []))
        if len(body) < 54:
            people.append(person)
            continue
        occluders = parts.get("eyepatch", []) + parts.get("glasses", []) + parts.get("mask", [])
        clean = {key: [p for p in parts.get(key, []) if key == "face" or not _occluded(p, occluders)]
                 for key in ("face", "eye", "closed_eye", "eyebrow", "nose", "mouth", "closed_mouth", "ear")}
        clean["eye"] = _merge_eyes(clean["eye"], clean["closed_eye"])
        clean["mouth"] = _merge_mouth(clean["mouth"], clean["closed_mouth"])
        if clean["mouth"]:
            mouth = clean["mouth"][0]
            mouth_width = mouth["box"][2] - mouth["box"][0]
            clean["nose"] = [nose for nose in clean["nose"]
                             if np.hypot(nose["x"]-mouth["x"], nose["y"]-mouth["y"])
                             > max(5.0, mouth_width * .2)]
        clean["ear"] = _visible_ear_candidates(clean)
        clean["eyebrow"] = _select_or_infer_brows(clean)
        # A rear head may contain ears, but no reliable front-facing evidence.
        nose_mouth = bool(clean["nose"] and clean["mouth"]
                          and clean["nose"][0]["y"] < clean["mouth"][0]["y"])
        front = bool(len(clean["eye"]) >= 2
                     or (clean["eye"] and (clean["nose"] or clean["mouth"] or clean["eyebrow"] or clean["face"] or clean["ear"] or occluders))
                     or nose_mouth)
        if not front:
            for idx in (0, 14, 15, 16, 17):
                body[idx*3+2] = 0.0
            person["pose_keypoints_2d"] = body
            person["face_keypoints_2d"] = [0.0] * 210
            people.append(person)
            continue
        def set_point(idx, p, score_scale=1.0):
            body[idx*3:idx*3+3] = [float(p["x"]), float(p["y"]), float(p["score"] * score_scale)]
        for idx in (0, 14, 15, 16, 17):
            body[idx*3+2] = 0.0
        if clean["nose"]:
            set_point(0, clean["nose"][0])
        for name, indices in (("eye", (14, 15)), ("ear", (16, 17))):
            candidates = clean[name][:4]
            if len(candidates) >= 2:
                first, second = candidates[:2]
                direct = sum((p["x"]-body[idx*3])**2 + (p["y"]-body[idx*3+1])**2
                             for p, idx in ((first, indices[0]), (second, indices[1])))
                swapped = sum((p["x"]-body[idx*3])**2 + (p["y"]-body[idx*3+1])**2
                              for p, idx in ((second, indices[0]), (first, indices[1])))
                ordered = (second, first) if swapped < direct else (first, second)
                for idx, p in zip(indices, ordered):
                    set_point(idx, p)
                continue
            # Match the detector's points to SAM3D's left/right 3D bone projection.
            for p in candidates:
                available = [idx for idx in indices if body[idx*3+2] <= 0]
                if not available:
                    break
                idx = min(available, key=lambda i: (p["x"]-float(body[i*3]))**2 + (p["y"]-float(body[i*3+1]))**2)
                set_point(idx, p)
        # Keep the 3D model's projection only where an observed occluder covers it.
        # This gives a usable bone under an eye patch, reflective lenses, or a mask.
        occlusion_for = {0: parts.get("mask", []),
                         14: parts.get("eyepatch", []) + parts.get("glasses", []),
                         15: parts.get("eyepatch", []) + parts.get("glasses", [])}
        for idx, covers in occlusion_for.items():
            if body[idx*3+2] > 0:
                continue
            x, y = body[idx*3:idx*3+2]
            if np.isfinite([x, y]).all() and _occluded({"x": x, "y": y}, covers):
                set_point(idx, {"x": x, "y": y, "score": .25})
        body = _infer_missing_nose(body, clean)
        body = _infer_missing_ears(body, clean)
        person["pose_keypoints_2d"] = body
        person["face_keypoints_2d"] = _pseudo_face_keypoints(clean, body)
        people.append(person)
        faces.append({"parts": clean, "occluders": occluders})
        person["sam31_face_visibility"] = "front_or_side"
    out["people"] = people
    out["sam31_face_parts"] = {"backend": "ComfyUI SAM3.1", "faces": faces}
    out["face_keypoints"] = {"backend": "ComfyUI SAM3.1", "faces": [
        {"keypoints": [person["face_keypoints_2d"][i:i+3]
                       for i in range(0, 210, 3)
                       if person["face_keypoints_2d"][i+2] > 0]}
        for person in people if len(person.get("face_keypoints_2d", [])) == 210
    ]}
    return out


def match_body_shape_head_to_pose(body_shape_pose, pose):
    """Keep measured head geometry when body shape is used as a render target."""
    people = []
    for target, source in zip(body_shape_pose.get("people", []), pose.get("people", [])):
        target = dict(target)
        target_points = list(target.get("pose_keypoints_2d", []))
        source_points = source.get("pose_keypoints_2d", [])
        if len(target_points) >= 54 and len(source_points) >= 54:
            for index in (0, 14, 15, 16, 17):
                offset = index * 3
                target_points[offset:offset + 3] = source_points[offset:offset + 3]
            target["pose_keypoints_2d"] = target_points
        people.append(target)
    people.extend(body_shape_pose.get("people", [])[len(people):])
    result = dict(body_shape_pose)
    result["people"] = people
    return result


def _torso_length(person):
    body = person.get("pose_keypoints_2d", [])
    if len(body) < 54 or any(body[index * 3 + 2] <= 0 for index in (1, 8, 11)):
        return None
    neck = np.asarray(body[3:5], dtype=float)
    hips = (np.asarray(body[8 * 3:8 * 3 + 2], dtype=float)
            + np.asarray(body[11 * 3:11 * 3 + 2], dtype=float)) / 2
    length = float(np.linalg.norm(neck - hips))
    return length if np.isfinite(length) and length > 1 else None


def _face_dimensions(person):
    face = person.get("face_keypoints_2d", [])
    if len(face) != 210:
        return None
    horizontal, vertical = _face_axes(person)
    if all(face[index * 3 + 2] > 0 for index in (0, 8, 16)):
        upper = [np.asarray(face[index * 3:index * 3 + 2], dtype=float)
                 for index in range(17, 48) if face[index * 3 + 2] > 0]
        if upper:
            width = abs(float(np.dot(np.asarray(face[16 * 3:16 * 3 + 2]) - face[0:2], horizontal)))
            height = abs(float(np.dot(np.asarray(face[8 * 3:8 * 3 + 2]) - np.mean(upper, axis=0), vertical)))
            if width > 2 and height > 2:
                return width, height, "jaw"
    if all(face[index * 3 + 2] > 0 for index in (68, 69, 51)):
        eyes = (np.asarray(face[68 * 3:68 * 3 + 2]) + np.asarray(face[69 * 3:69 * 3 + 2])) / 2
        width = abs(float(np.dot(np.asarray(face[69 * 3:69 * 3 + 2]) - face[68 * 3:68 * 3 + 2], horizontal)))
        height = abs(float(np.dot(np.asarray(face[51 * 3:51 * 3 + 2]) - eyes, vertical)))
        if width > 2 and height > 2:
            return width, height, "eyes_mouth"
    return None


def _face_axes(person):
    body = person.get("pose_keypoints_2d", [])
    if len(body) >= 54 and all(body[index * 3 + 2] > 0 for index in (14, 15)):
        horizontal = np.asarray(body[15 * 3:15 * 3 + 2], dtype=float) - body[14 * 3:14 * 3 + 2]
        length = np.linalg.norm(horizontal)
        if length > 2:
            horizontal /= length
            return horizontal, np.array([-horizontal[1], horizontal[0]])
    return np.array([1.0, 0.0]), np.array([0.0, 1.0])


def adapt_pose_face_to_body_shape(pose, body_shape_face_people):
    """Transfer face-to-torso proportions while keeping pose and expression."""
    adjusted = []
    changed = False
    for source, target in zip(pose.get("people", []), body_shape_face_people):
        source_size, target_size = _face_dimensions(source), _face_dimensions(target)
        source_torso, target_torso = _torso_length(source), _torso_length(target)
        if not (source_size and target_size and source_torso and target_torso) or source_size[2] != target_size[2]:
            adjusted.append(source)
            continue
        scale_x = float(np.clip((target_size[0] / target_torso) / (source_size[0] / source_torso), .65, 1.6))
        scale_y = float(np.clip((target_size[1] / target_torso) / (source_size[1] / source_torso), .65, 1.6))
        face = list(source["face_keypoints_2d"])
        anchor_index = next((index for index in (30, 33, 36, 42, 8) if face[index * 3 + 2] > 0), None)
        if anchor_index is None:
            adjusted.append(source)
            continue
        anchor_x, anchor_y = face[anchor_index * 3:anchor_index * 3 + 2]
        horizontal, vertical = _face_axes(source)
        anchor = np.array([anchor_x, anchor_y])
        for index in range(70):
            offset = index * 3
            if face[offset + 2] > 0:
                displacement = np.asarray(face[offset:offset + 2]) - anchor
                face[offset:offset + 2] = (anchor + horizontal * np.dot(displacement, horizontal) * scale_x
                                          + vertical * np.dot(displacement, vertical) * scale_y).tolist()
        body = list(source["pose_keypoints_2d"])
        for index in (0, 14, 15, 16, 17):
            offset = index * 3
            if body[offset + 2] > 0:
                displacement = np.asarray(body[offset:offset + 2]) - anchor
                body[offset:offset + 2] = (anchor + horizontal * np.dot(displacement, horizontal) * scale_x
                                          + vertical * np.dot(displacement, vertical) * scale_y).tolist()
        person = dict(source)
        person["face_keypoints_2d"] = face
        person["pose_keypoints_2d"] = body
        adjusted.append(person)
        changed = True
    adjusted.extend(pose.get("people", [])[len(adjusted):])
    if not changed:
        return pose
    result = dict(pose)
    result["people"] = adjusted
    if "face_keypoints" in pose:
        meta = dict(pose["face_keypoints"])
        meta["faces"] = [{"keypoints": [face[i:i + 3] for i in range(0, 210, 3) if face[i + 2] > 0]}
                         for person in adjusted
                         for face in [person.get("face_keypoints_2d", [])]
                         if len(face) == 210]
        result["face_keypoints"] = meta
    return result
