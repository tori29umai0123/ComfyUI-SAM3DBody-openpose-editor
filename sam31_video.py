"""Use ComfyUI's native SAM3.1 video tracker to keep person masks and IDs."""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .sam31_face import _text_embedding, load_model_and_clip, run_with_attention_fallback
except ImportError:
    from sam31_face import _text_embedding, load_model_and_clip, run_with_attention_fallback


def _track_data_for_images(images, threshold, max_objects, detect_interval):
    import comfy.model_management
    import comfy.utils

    model, clip = load_model_and_clip()
    if getattr(model.model.model_config, "unet_config", {}).get("image_model") != "SAM31":
        raise ValueError("Video tracking requires a SAM3.1 MODEL")
    network = model.model.diffusion_model
    if not hasattr(network, "forward_video"):
        raise RuntimeError("This ComfyUI version does not provide SAM3.1 video tracking")
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    embedding, text_mask = _text_embedding(clip, "person", device, dtype)
    frames = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
    frames = frames[..., :3].movedim(-1, 1)
    with torch.inference_mode():
        result = run_with_attention_fallback(lambda: network.forward_video(
            images=frames, initial_masks=None,
            pbar=comfy.utils.ProgressBar(int(frames.shape[0])),
            text_prompts=[(embedding, text_mask)], new_det_thresh=float(threshold),
            max_objects=int(max_objects), detect_interval=int(detect_interval),
            target_device=device, target_dtype=dtype))
    result["orig_size"] = (int(images.shape[1]), int(images.shape[2]))
    return result


def regions_from_track_data(track_data, frame_count, height, width):
    """Return visible person masks with IDs assigned left to right on first appearance."""
    from comfy.ldm.sam3.tracker import unpack_masks

    if not isinstance(track_data, dict) or int(track_data.get("n_frames", -1)) != frame_count:
        raise ValueError("SAM3 track data must have the same number of frames as pose_image")
    original_size = track_data.get("orig_size")
    if original_size is not None and tuple(map(int, original_size)) != (height, width):
        raise ValueError("SAM3 track data must have the same frame size as pose_image")
    packed = track_data.get("packed_masks")
    if packed is None:
        return [[] for _ in range(frame_count)]
    if packed.ndim != 4 or packed.shape[0] != frame_count:
        raise ValueError("Invalid SAM3 packed_masks shape")
    track_to_slot = {}
    frames = []
    scores = track_data.get("scores")
    if scores is None:
        scores = []
    for frame_index in range(frame_count):
        unpacked = unpack_masks(packed[frame_index:frame_index + 1])
        resized = F.interpolate(unpacked.float(), size=(height, width), mode="bilinear", align_corners=False)
        masks = (resized[0] > 0.5).cpu().numpy()
        visible = []
        for track_index, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if len(xs) < 100:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            visible.append({"box": [x1, y1, x2, y2], "mask": mask,
                            "score": float(scores[track_index]) if track_index < len(scores) else 1.0,
                            "sam3_track_index": track_index})
        newcomers = sorted((region for region in visible if region["sam3_track_index"] not in track_to_slot),
                           key=lambda region: (region["box"][0], region["box"][1]))
        for region in newcomers:
            track_to_slot[region["sam3_track_index"]] = len(track_to_slot)
        for region in visible:
            region["track_id"] = track_to_slot[region["sam3_track_index"]]
        frames.append(sorted(visible, key=lambda region: region["track_id"]))
    return frames


def track_people_video(images, threshold=0.5, max_objects=16, detect_interval=1, track_data=None):
    shape = images.shape if hasattr(images, "shape") else np.asarray(images).shape
    if len(shape) != 4:
        raise ValueError("Video Analyze expects an IMAGE batch [frames, height, width, channels]")
    count, height, width = map(int, shape[:3])
    if count < 1:
        raise ValueError("Video Analyze requires at least one frame")
    if track_data is None:
        track_data = _track_data_for_images(images, threshold, max_objects, detect_interval)
    return regions_from_track_data(track_data, count, height, width)
