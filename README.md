# ComfyUI-SAM3DBody-openpose-editor

SAM 3D Body inference to OpenPose-compatible `POSE_KEYPOINT` data.

## Nodes

- `SAM3D Body OpenPose Analyze`
  - Inputs: required `pose_image`, optional `body_shape_image`, `device`, `auto_head_proportion`, `head_confidence`
  - Outputs: `pose_keypoint`, `body_shape_keypoint`, `debug_person_crops`
  - With `auto_head_proportion` enabled, the body-shape output keeps the head keypoints measured from `pose_image`, so Render does not shrink the detected head to SAM3DBody's default size. When `body_shape_image` is connected, SAM3.1 also measures its face and adjusts the pose face width and height relative to torso length. Face direction and expression stay with `pose_image`; if the shape face is not visible, the pose face is kept.
  - RTMDet-Ins is used for person boxes, and AnimeHeads/anime-face-detector are used internally for face IDs, head direction, and head-size hints. If a detector fails, the node falls back to the previous SAM3DBody-only behavior where possible.
  - Multi-person images are supported. People are sorted from top-left to bottom-right. `body_shape_image` may be a single image or an image batch; detected body shapes are assigned in the same top-left order. Missing body-shape entries fall back to the corresponding person from `pose_image`; extra body-shape entries are ignored.
- `SAM3D Body OpenPose Render`
  - Inputs: required `pose_keypoint`, optional `body_shape_keypoint`
  - Outputs: rendered OpenPose image

`pose_keypoint` uses COCO-18 body order plus OpenPose 21-point hands.
`body_shape_keypoint` includes the same OpenPose structure plus raw SAM3DBody MHR70 / 3D / parameter data for downstream retargeting.

## Sample workflow

Load [`workflows/sam3d_body_openpose_example.json`](workflows/sam3d_body_openpose_example.json) in ComfyUI, select an image in `Load Image`, then queue the workflow. It runs Analyze → Render → Save Image. To use a separate body-shape reference, connect another `Load Image` node to Analyze's `body_shape_image` input and connect Analyze's `body_shape_keypoint` output to Render's matching input. The node uses `models/sam3.1/sam3.1_multiplex_fp16.safetensors` if present and downloads it automatically on first use.

## Models

Analyze looks for these files under ComfyUI's configured models directory and downloads them automatically when missing:

- `models/sam3dbody/model.ckpt` and `models/sam3dbody/assets/mhr_model.pt`
  - Source: https://huggingface.co/jetjodh/sam-3d-body-dinov3
- `models/animeheads/animeheadsv3.onnx`
  - Source: https://huggingface.co/tori29umai/AnimeHeads-ONNX
- `models/rtdetrv4/rtdetrv4-x-manga109s_v2.onnx`
  - Source: https://huggingface.co/tori29umai/rtdetrv4-x-manga109s_v2

AnimeHeads is used for face IDs, back-facing head support, and head-size estimation. RT-DETRv4 is used for person boxes. If a Hugging Face download fails with Windows error 10054, the node retries that download over IPv4 in a separate process. Other download errors are reported normally.

## anime-face-detector

Analyze uses ComfyUI's native SAM3.1 inference to locate the face, eyes, eyebrows, nose, mouth and ears around each SAM3D head. It reuses `models/sam3.1/sam3.1_multiplex_fp16.safetensors` when present and otherwise downloads it from [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1). Detected parts are converted into an approximate 70-slot OpenPose `face_keypoints_2d` array; slots for parts that cannot be seen or detected have zero confidence. Render uses the detected face parts for the head bones without drawing the detailed landmark points. The jaw follows the detected face mask outline, and eye and mouth points follow their visible masks in the tilted face coordinate frame. Eyebrow detections that resemble eyelashes are rejected; missing eyebrow points are estimated from the eyes and the face tilt. Thin eye or mouth masks are treated as closed, hiding pupil or inner-mouth points. Ear detections are used only at plausible positions beside the eyes; hidden ears are estimated at low confidence from the eye line and nose or mouth direction so the head bones remain visible without connecting to hair accessories. A nose detection that overlaps the mouth is ignored; its head anchor is estimated from the eyes and mouth. These are geometric estimates, not 70 separately detected landmarks. The detected eye/nose/ear coordinates also update the OpenPose face bones. On a back view, unsupported facial bones are hidden. An eye patch, glasses or face mask can preserve covered SAM3D bone projections at low confidence. SAM3.1 detections are heuristic and should be checked on difficult art.

For images with multiple people, Analyze matches SAM3.1 `person` boxes and masks to the SAM3DBody person boxes, then runs face-part detection on a separate masked head crop for each person. If person segmentation fails, it continues with the original head crops. If SAM3.1 is unavailable, `anime-face-detector` remains the fallback for anime face landmarks and head-size/head-point hints. There is no Analyze UI selector for face backends.

`install.py` uses environment-specific rules:

- Windows / Linux on Python 3.12 or newer: install the upstream package from PyPI when it supports that interpreter.
- Windows on Python 3.10 or 3.11: download the wheel from `tori29umai/anime_face_detector_Wheels` on Hugging Face and install it.
- Other environments: download the upstream source, patch `requires-python` for the current Python version, and build/install locally.
