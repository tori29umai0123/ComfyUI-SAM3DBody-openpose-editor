"""Keep per-person reference slots stable across a sequence of image frames."""

from functools import lru_cache

import cv2
import numpy as np


def _iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return overlap / max(1.0, area_a + area_b - overlap)


def _center(box):
    return (box[:2] + box[2:]) * 0.5


def _features(image, outputs, candidates):
    """Match SAM3D boxes to SAM masks, then describe the visible person's colors."""
    boxes = []
    for output in outputs:
        value = output.get("bbox")
        box = np.asarray(value, dtype=np.float32).reshape(-1)[:4] if value is not None else np.array([0, 0, 0, 0], dtype=np.float32)
        boxes.append(box)
    pairs = sorted((( _iou(box, np.asarray(candidate["box"], dtype=np.float32)), i, j)
                    for i, box in enumerate(boxes) for j, candidate in enumerate(candidates)), reverse=True)
    matched, used = {}, set()
    for score, i, j in pairs:
        if score > 0 and i not in matched and j not in used:
            matched[i] = candidates[j]
            used.add(j)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features = []
    for i, box in enumerate(boxes):
        candidate = matched.get(i)
        if candidate is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = box.astype(int)
            mask[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)] = 1
        else:
            mask = np.asarray(candidate["mask"], dtype=np.uint8)
        color = cv2.calcHist([hsv], [0, 1], mask, [12, 8], [0, 180, 0, 256]).reshape(-1)
        light = cv2.calcHist([hsv], [2], mask, [16], [0, 256]).reshape(-1)
        color /= max(1.0, float(color.sum()))
        light /= max(1.0, float(light.sum()))
        features.append((box, np.concatenate((color * 0.7, light * 0.3))))
    return features


class PersonTracker:
    """Assign persistent IDs; the first frame uses the incoming top-left order."""

    def __init__(self, max_missing_frames=12):
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.tracks = {}
        self.next_id = 0

    def update(self, image, outputs, candidates, frame_index):
        detections = _features(image, outputs, candidates)
        active = [(id_, track) for id_, track in self.tracks.items()
                  if frame_index - track["frame"] <= self.max_missing_frames + 1]
        costs = []
        for box, hist in detections:
            row = []
            for _, track in active:
                gap = frame_index - track["frame"]
                predicted = track["box"] + np.tile(track["velocity"] * gap, 2)
                size = max(1.0, np.linalg.norm(track["box"][2:] - track["box"][:2]))
                distance = np.linalg.norm(_center(box) - _center(predicted)) / size
                appearance = 1.0 - np.minimum(hist, track["hist"]).sum()
                scale = abs(np.log(max(1.0, np.prod(box[2:] - box[:2])) /
                                   max(1.0, np.prod(track["box"][2:] - track["box"][:2]))))
                cost = 1.7 * distance + 1.5 * appearance + 0.35 * scale - 0.4 * _iou(box, predicted)
                row.append(float(cost) if distance < 2.0 + 0.5 * gap else float("inf"))
            costs.append(row)

        if len(active) <= 14 and len(detections) <= 14:
            @lru_cache(None)
            def assign(index, used):
                if index == len(detections):
                    return 0.0, ()
                best_cost, tail = assign(index + 1, used)
                best_cost += 2.0  # Starting a new track has a cost.
                best = (best_cost, (None,) + tail)
                for j, cost in enumerate(costs[index]):
                    if used & (1 << j) or cost >= 2.0:
                        continue
                    rest, tail = assign(index + 1, used | (1 << j))
                    if cost + rest < best[0]:
                        best = (cost + rest, (j,) + tail)
                return best

            matches = assign(0, 0)[1]
        else:
            matches = [None] * len(detections)
            used = set()
            for cost, i, j in sorted((cost, i, j) for i, row in enumerate(costs) for j, cost in enumerate(row)):
                if cost >= 2.0:
                    break
                if matches[i] is None and j not in used:
                    matches[i] = j
                    used.add(j)

        ids = []
        for (box, hist), match in zip(detections, matches):
            if match is None:
                id_ = self.next_id
                self.next_id += 1
                velocity = np.zeros(2, dtype=np.float32)
            else:
                id_, previous = active[match]
                gap = max(1, frame_index - previous["frame"])
                velocity = (_center(box) - _center(previous["box"])) / gap
                hist = 0.6 * hist + 0.4 * previous["hist"]
            self.tracks[id_] = {"box": box, "hist": hist, "velocity": velocity, "frame": frame_index}
            ids.append(id_)
        return ids
