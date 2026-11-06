# ComfyUI-SAM3DBody-openpose-editor

SAM 3D Body inference to OpenPose-compatible `POSE_KEYPOINT` data.

## Nodes

- `SAM3D Body OpenPose Analyze`
  - Inputs: required `pose_image`, optional `body_shape_image` or `body_shape_images`, `device`, `auto_head_proportion`, `person_threshold`
  - Outputs: `pose_keypoint`, `body_shape_keypoint`, `debug_person_crops`
  - With `auto_head_proportion` enabled and a body-shape image connected, SAM3.1 measures face width in both images. The reference's face-to-torso ratio is transferred per person: a chibi reference shortens the body and enlarges the pose head up to 2×. The pose's face direction and expression stay in place. When the reference face cannot be measured, body lengths are normalized by torso length.
  - SAM3.1 detects each person's box and mask. SAM3DBody estimates the pose within each mask, and SAM3.1 detects face parts.
  - Multi-person images are supported. People are sorted by the top-left position of their detected bbox. Reference body shapes are assigned in that order. A single `body_shape_image`, an image batch, and a chained `body_shape_images` collection are supported. If both inputs are connected, `body_shape_image` comes first. Missing or failed reference slots use the corresponding person from `pose_image`; extra references are ignored.
  - Render applies each reference person's body proportions separately. Reference keypoints are normalized to the pose person's scale and head points are aligned to that reference person's neck before measuring lengths.
- `SAM3D Body Shape Images (Append)`
  - Connect one `IMAGE` to `body_shape_image`. Connect a previous Append node's `body_shape_images` output to this node's `body_shape_images` input to add another reference. Chain as many Append nodes as needed, then connect the last output to Analyze's `body_shape_images` input. Each image batch contributes its frames in batch order.
- `SAM3D Body OpenPose Render`
  - Inputs: required `pose_keypoint`, optional `body_shape_keypoint`
  - Outputs: rendered OpenPose image

`pose_keypoint` uses COCO-18 body order plus OpenPose 21-point hands.
`body_shape_keypoint` includes the same OpenPose structure plus raw SAM3DBody MHR70 / 3D / parameter data for downstream retargeting.

## Sample workflow

Load [`workflows/sam3d_body_openpose_example.json`](workflows/sam3d_body_openpose_example.json) in ComfyUI, select an image in `Load Image`, then queue the workflow. It runs Analyze → Render → Save Image. For one body-shape reference, connect another `Load Image` node to Analyze's `body_shape_image` input. For multiple references, chain `SAM3D Body Shape Images (Append)` nodes in person order and connect the last output to Analyze's `body_shape_images` input. Connect Analyze's `body_shape_keypoint` output to Render's matching input. The node uses `models/sam3.1/sam3.1_multiplex_fp16.safetensors` if present and downloads it automatically on first use.

## Models

Analyze looks for these files under ComfyUI's configured models directory and downloads them automatically when missing:

- `models/sam3dbody/model.ckpt` and `models/sam3dbody/assets/mhr_model.pt`
  - Source: https://huggingface.co/jetjodh/sam-3d-body-dinov3
- `models/sam3.1/sam3.1_multiplex_fp16.safetensors`
  - Source: https://huggingface.co/Comfy-Org/sam3.1

If a Hugging Face download fails with Windows error 10054, the node retries that download over IPv4 in a separate process. Other download errors are reported normally.

## SAM3.1 face detection

Analyze uses ComfyUI's native SAM3.1 inference to locate the face, eyes, eyebrows, nose, mouth and ears around each SAM3D head. It reuses `models/sam3.1/sam3.1_multiplex_fp16.safetensors` when present and otherwise downloads it from [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1). Detected parts are converted into an approximate 70-slot OpenPose `face_keypoints_2d` array; slots for parts that cannot be seen or detected have zero confidence. Render uses the detected face parts for the head bones without drawing the detailed landmark points. The jaw follows the detected face mask outline, and eye and mouth points follow their visible masks in the tilted face coordinate frame. Eyebrow detections that resemble eyelashes are rejected; missing eyebrow points are estimated from the eyes and the face tilt. Thin eye or mouth masks are treated as closed, hiding pupil or inner-mouth points. Ear detections are used only at plausible positions beside the eyes; hidden ears are estimated at low confidence from the eye line and nose or mouth direction so the head bones remain visible without connecting to hair accessories. A nose detection that overlaps the mouth is ignored; its head anchor is estimated from the eyes and mouth. These are geometric estimates, not 70 separately detected landmarks. The detected eye/nose/ear coordinates also update the OpenPose face bones. On a back view, unsupported facial bones are hidden. An eye patch, glasses or face mask can preserve covered SAM3D bone projections at low confidence. SAM3.1 detections are heuristic and should be checked on difficult art.

For images with multiple people, Analyze reuses the SAM3.1 person masks to isolate each head crop during face-part detection. If SAM3.1 finds no people, Analyze reports an error instead of silently dropping people.

On GPUs unsupported by the installed xFormers build, SAM3.1 retries with PyTorch attention automatically.
