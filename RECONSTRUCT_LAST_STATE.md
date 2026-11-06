# 最後の実装状態を再現する設計書

この文書は「安全に作り直す設計」ではなく、会話ログ上で最後に到達していた `nodes.py` の状態をベースに、追加指示で修正された仕様を再現するための設計書である。

参照元の実体は `nodes_failed_sam31_attempt_20260724.py`。このファイルは構文チェックと import は通るが、ユーザー評価としては「全然ダメ」だった状態である。この文書では、その状態から OpenPose顔点削除と白点だけ描画を修正する。

## 再現対象

再現対象は次の状態。

- `SAM3D Body OpenPose Analyze` / `SAM3D Body OpenPose Render` の2ノード構成
- Analyze UI から `model_path` は削除済み
- SAM3D Body model は `ComfyUI/models/sam3dbody` 決め打ち、自動DL
- AnimeHeads ONNX を使う
- RT-DETRv4 manga109s body bbox を使う
- head count を人物数の起点にする
- head と body bbox を対応付ける
- body bbox を SAM3/SAM3.1風 bbox prompt に渡して mask を作ろうとする
- 取得した mask を SAM3DBodyEstimator の `masks=` に渡す
- face detector は guidance の head bbox を優先して誘導する
- OpenPose式の鼻・目・耳点は可視顔判定で消さない
- Render で anime-face-detector の顔keypointを線と点で追加描画する。鼻は点のみ

## 重要な注意

この最後の状態では、コード上の名称は `sam3.1` だが、実際の loader は `sam3` Python package の `build_sam3_image_model` と `Sam3Processor` を使う。

`C:\ComfyUI\models\sam3.1\sam3.1_multiplex_fp16.safetensors` を直接読む実装ではない。

実際に優先する checkpoint は次。

```python
SAM3_LOCAL_CHECKPOINT_PATH = os.path.join(DEFAULT_MODELS_DIR, "sam3", "sam3.pt")
```

存在すれば `C:\ComfyUI\models\sam3\sam3.pt` を使う。なければ `facebook/sam3` からHF経由でDLを試みる。

## モデル定数

```python
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "sam3dbody")
ANIMEHEADS_MODEL_RELATIVE_PATH = os.path.join("animeheads", "animeheadsv3.onnx")
RTDETRV4_BODY_MODEL_RELATIVE_PATH = os.path.join("rtdetrv4", "rtdetrv4-x-manga109s_v2.onnx")
RTDETRV4_BODY_REPO_ID = "tori29umai/rtdetrv4-x-manga109s_v2"
RTDETRV4_BODY_REPO_FILENAME = "model.onnx"
RTDETRV4_BODY_SCORE_THRESHOLD = 0.10
SAM3_LOCAL_CHECKPOINT_PATH = os.path.join(DEFAULT_MODELS_DIR, "sam3", "sam3.pt")
```

キャッシュ:

```python
_SAM3D_MODEL_CACHE = {}
_ANIMEHEADS_NET_CACHE = {}
_RTDETRV4_BODY_SESSION_CACHE = {}
_SAM31_PROCESSOR_CACHE = {}
_ANIME_FACE_DETECTOR_CACHE = {}
```

## 処理全体の流れ

Analyze は以下の順で処理する。

1. `pose_image` を RGB/BGR numpy に変換
2. AnimeHeads で head bbox を検出
3. head がなければ guided bbox/mask は使わず、SAM3DBody full image fallback
4. RT-DETRv4 で body bbox を検出
5. 各 head に最も近い body bbox を割り当てる
6. body bbox が見つからない head は head から推定 body bbox を作る
7. bbox rectangle mask を仮作成し、`mask_source = "bbox-rectangle-pending-sam3.1"` とする
8. bbox list を SAM bbox prompt に投げ、mask 候補を取得する
9. SAM mask が取れた bbox は mask を差し替え、`mask_source = "sam3.1"` にする
10. SAM3DBodyEstimator に `bboxes=bboxes, masks=masks` を渡す
11. SAM3DBody出力に `sam3dbody_guidance` を埋める
12. OpenPose形式に変換
13. anime-face-detector で顔keypointを注釈する
14. OpenPose本体の nose/eye/ear は削除・非表示化しない
15. Render で OpenPoseを描画し、anime-face-detector の顔keypointを線と点で追加描画する。鼻は点のみ

## head/body/mask guidance

`_body_boxes_from_heads_and_rtdetr(image_bgr, head_confidence, device)` は3つ返す。

```python
return bboxes, targets, masks
```

- `bboxes`: `np.float32 [N, 4]`
- `targets`: head/body/mask対応metadata list
- `masks`: `np.uint8 [N, H, W]`

`targets` の要素:

```json
{
  "head_id": 0,
  "head_bbox": [x1, y1, x2, y2],
  "head_confidence": 0.9,
  "body_bbox": [x1, y1, x2, y2],
  "body_source": "rtdetrv4-x-manga109s_v2",
  "body_confidence": 0.8,
  "mask_source": "sam3.1",
  "mask_bbox": [x1, y1, x2, y2],
  "mask_confidence": 0.0,
  "mask_assignment_score": 0.5,
  "mask_area": 12345
}
```

body bbox fallback時:

```json
{
  "body_source": "head-estimate",
  "body_confidence": null
}
```

SAM mask取得前または失敗時:

```json
{
  "mask_source": "bbox-rectangle-pending-sam3.1",
  "mask_confidence": 0.0,
  "mask_assignment_score": 0.0
}
```

## head と body bbox の対応付け

`_target_box_assignment_score(head, person_box)` を使う。

スコア要素:

- head center が body bbox 内なら +4.0
- head size と body bbox短辺の比率を少し加点
- bbox center から head center までの正規化距離を減点

採用条件:

```python
best_i is not None and best_s >= -0.15
```

未採用時は head bbox から body bbox を推定する。

```python
box = [cx - hw * 2.4, y1 - hh * 0.9, cx + hw * 2.4, y2 + hh * 6.0]
```

## SAM bbox prompt mask

関数構成:

- `_normalize_sam31_device(device)`
- `_load_sam31_processor(device="cuda")`
- `_box_xyxy_to_sam31_cxcywh(box, width, height)`
- `_sam31_state_masks_to_candidates(state, prompt_box)`
- `_detect_sam31_person_masks_from_boxes(image_rgb, boxes, device="cuda")`

### loader

```python
def _load_sam31_processor(device="cuda"):
    sam_device = _normalize_sam31_device(device)
    key = (sam_device, 1008)
    if key not in _SAM31_PROCESSOR_CACHE:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        checkpoint_path = SAM3_LOCAL_CHECKPOINT_PATH if os.path.exists(SAM3_LOCAL_CHECKPOINT_PATH) else None
        model = build_sam3_image_model(
            device=sam_device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        _SAM31_PROCESSOR_CACHE[key] = Sam3Processor(
            model,
            resolution=1008,
            device=sam_device,
            confidence_threshold=0.0,
        )
    return _SAM31_PROCESSOR_CACHE[key]
```

### bbox変換

SAM prompt は正規化 `[center_x, center_y, width, height]`。

```python
[x_center / image_width, y_center / image_height, box_width / image_width, box_height / image_height]
```

### mask候補選択

`processor.add_geometric_prompt(prompt, True, state)` の結果から:

- `state["masks"]`
- `state["boxes"]`
- `state["scores"]`

を取り出す。

候補は `prompt_iou`, `score`, `area` の順で降順ソートし、先頭を採用する。

```python
candidates.sort(
    key=lambda c: (
        float(c.get("prompt_iou", 0.0)),
        float(c.get("score", 0.0)),
        int(c.get("area", 0)),
    ),
    reverse=True,
)
```

## SAM3DBodyEstimator 呼び出し

`_run_sam3d_estimator()` はまず guided bbox/mask を作る。

```python
bboxes, targets, masks = _body_boxes_from_heads_and_rtdetr(img_bgr, head_confidence, device)
```

その後:

```python
outputs = estimator.process_one_image(
    img_rgb,
    bboxes=bboxes,
    masks=masks,
    bbox_thr=bbox_threshold,
    inference_type=inference_type,
)
```

fallback順:

1. `bboxes + masks`
2. `bboxes` only
3. full image

```python
if not outputs:
    outputs = estimator.process_one_image(img_rgb, bboxes=bboxes, bbox_thr=bbox_threshold, inference_type=inference_type)
if not outputs:
    outputs = estimator.process_one_image(img_rgb, bboxes=None, bbox_thr=bbox_threshold, inference_type=inference_type)
    targets = []
```

SAM3DBody出力に `targets[idx]` を入れる。

```python
output["sam3dbody_guidance"] = targets[idx]
```

## pose_keypoint metadata

`_make_openpose_keypoint_from_outputs()` で以下を追加する。

```python
guidance = [_to_jsonable(o.get("sam3dbody_guidance")) for o in outputs]
if any(g is not None for g in guidance):
    pose["sam3dbody_guidance"] = guidance
```

`_make_sam3dbody_metadata()` には以下を含める。

```python
"bbox": _to_jsonable(output.get("bbox")),
"guidance": _to_jsonable(output.get("sam3dbody_guidance")),
```

## anime-face-detector 誘導

顔検出は `_detect_anime_face_keypoints()`。

流れ:

1. `pose_keypoint["sam3dbody_guidance"]` から head bbox を取り出す
2. guidance があればそれを heads とする
3. guidance がなければ AnimeHeads を再検出する
4. 全体画像で通常 anime-face-detector 検出
5. 各 head bbox crop を拡大し、`detector(detector_input, boxes=[box])` で誘導検出
6. `_filter_faces_by_heads()` で head ごとに1顔へ絞る
7. `_drop_unconfirmed_guided_faces()` で誘導のみの顔を落とす

### `_heads_from_pose_guidance()`

`sam3dbody_guidance` の `head_bbox` を次の形式に変換する。

```python
{
  "bbox": [x1, y1, x2, y2],
  "confidence": head_confidence,
  "face_id": i,
  "source": "sam3dbody-guidance"
}
```

### `_drop_unconfirmed_guided_faces()`

`source == "animeheads-guided"` の顔は、通常検出顔と近い場合だけ残す。

通常検出顔が1つもない場合は空listを返す。

```python
if not confirmed:
    return []
```

近接条件:

```python
np.hypot(fcx - ocx, fcy - ocy) <= max(18.0, min(fsize, osize) * 0.65)
```

## OpenPose顔点の扱い

対象OpenPose body index:

- `0`: nose
- `14`: right eye
- `15`: left eye
- `16`: right ear
- `17`: left ear

この5点は、顔の可視・不可視判定では削除しない。

仕様:

- SAM3DBody から出た OpenPose body keypoint をそのまま残す
- 後ろ向き、横向き、顔面が見えない人物でも、SAM3DBody が出した nose/eye/ear の confidence を `0.0` にしない
- anime-face-detector の結果がないことを理由に OpenPose body 顔点を消さない
- `_remove_hidden_face_landmarks()` は Analyze の処理順から外す
- `_hide_invisible_openpose_face_parts()` は再現実装では呼ばない。残す場合も互換用の未使用関数にする

顔の可視判定や `_drop_unconfirmed_guided_faces()` は、anime-face-detector 由来の詳細顔keypointを採用するかどうかだけに使う。OpenPose本体の鼻・目・耳点には影響させない。

## Render

Render は `ComfyUI-ultimate-openpose-editor/util.py` の `draw_pose_json()` を呼ぶ。

その後、`pose_keypoint["face_keypoints"]["faces"]` にある anime-face-detector keypoint を線と点で上描きする。

関数:

```python
_draw_face_keypoints_on_images(...)
```

描画仕様:

- `face_keypoint_size` が `0` 以下なら描かない
- confidence が `0` より大きい keypoint だけ点を描く
- 線は両端keypointの confidence が `0` より大きい場合だけ描く
- 点と線は白で描く
- 線幅は `max(1, face_keypoint_size)` を基準にする
- 鼻 keypoint は点だけ描き、線ではつながない

68点顔keypointの線接続:

- 顔輪郭: `0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16`
- 右眉: `17-18-19-20-21`
- 左眉: `22-23-24-25-26`
- 右目: `36-37-38-39-40-41-36`
- 左目: `42-43-44-45-46-47-42`
- 外口: `48-49-50-51-52-53-54-55-56-57-58-59-48`
- 内口: `60-61-62-63-64-65-66-67-60`

鼻:

- `27-35` は点のみ
- `27-28-29-30` や `31-32-33-34-35` の線は描かない

## Analyze INPUT_TYPES

`model_path` は出さない。

最後の状態の required:

```python
{
  "pose_image": ("IMAGE",),
  "device": (["cuda", "cpu"], {"default": "cuda"}),
  "bbox_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
  "inference_type": (["full", "body", "hand"], {"default": "full"}),
  "confidence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
  "auto_head_proportion": ("BOOLEAN", {"default": True}),
  "head_confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
}
```

optional:

```python
{"body_shape_image": ("IMAGE",)}
```

## Render INPUT_TYPES

最後の状態では以下を含む。

- `show_body`
- `show_face`
- `show_hands`
- `face_keypoint_size`
- `width`
- `height`
- `offset_x`
- `offset_y`
- `camera_zoom`
- `use_ground_plane`
- `pose_marker_size`
- `face_marker_size`
- `hand_marker_size`
- `pelvis_scale`
- `torso_scale`
- `neck_scale`
- `head_scale`
- `eye_distance_scale`
- `left_eye_scale`
- `right_eye_scale`
- `left_eyebrow_scale`
- `right_eyebrow_scale`
- `mouth_scale`
- `nose_scale_face`
- `face_shape_scale`
- `shoulder_scale`
- `arm_scale`
- `leg_scale`
- `hands_scale`
- `overall_scale`

optional:

- `body_shape_keypoint`

## 自動DL対象

- SAM3DBody: `jetjodh/sam-3d-body-dinov3`
- AnimeHeads: `tori29umai/AnimeHeads-ONNX`, `weights/animeheadsv3.onnx`
- RT-DETRv4 body: `tori29umai/rtdetrv4-x-manga109s_v2`, `model.onnx`
- SAM/SAM3風 processor: `C:\ComfyUI\models\sam3\sam3.pt` がなければ `facebook/sam3`

## 既知の問題

この最後の状態は、ユーザー評価では「全然ダメ」。既知問題は以下。

1. `mask_source` は `sam3.1` と記録するが、実ロードは `sam3.pt` APIであり、本物の `sam3.1_multiplex_fp16.safetensors` ではない
2. SAM mask生成が重く、初回ロード/推論が120秒でタイムアウトした
3. bbox prompt mask と SAM3DBody人物出力の対応が index 順依存
4. `_sort_outputs_top_left()` 後に guidance の対応がずれる可能性がある
5. `body_shape_image` 側でも同じ heavy guidance flow が走る
6. 誘導顔検出を通常顔検出で確認するため、横顔・小顔が消えすぎる可能性がある
7. OpenPose顔点は削除しないため、SAM3DBody由来の不可視 nose/eye/ear が残る可能性がある
8. `auto_head_proportion` はUIにあるが、実質的に使われていない

## 再現手順

既存のまともな土台として `nodes_rebuild_working_backup.py` を使う場合、次の順で差分を入れる。

1. SAM3/SAM3.1風 bbox prompt helper を追加
2. `_body_boxes_from_heads_and_rtdetr()` の戻り値を `(bboxes, targets, masks)` に変える
3. headごとに body bbox と rectangle mask を作る
4. SAM prompt mask で rectangle mask を差し替える
5. `_run_sam3d_estimator()` で `masks=masks` を渡す
6. SAM3DBody出力に `sam3dbody_guidance` を埋める
7. `_make_openpose_keypoint_from_outputs()` に `sam3dbody_guidance` を出す
8. `_make_sam3dbody_metadata()` に `bbox` と `guidance` を追加
9. `_heads_from_pose_guidance()` を追加
10. `_detect_anime_face_keypoints()` を guidance head 優先に変える
11. `_drop_unconfirmed_guided_faces()` を追加して誘導顔を通常検出で確認する
12. `_remove_hidden_face_landmarks()` / `_hide_invisible_openpose_face_parts()` を Analyze から外し、OpenPose nose/eye/ear は消さない
13. `_draw_face_keypoints_on_images()` を白点だけではなく線+点描画に戻す。ただし鼻 `27-35` は点のみ
14. `_face_matches_pose_head()` の confidence 閾値を `0.65` にする
15. compile/import check を実行

## 検証コマンド

```powershell
C:\ComfyUI\.venv\Scripts\python.exe -m py_compile .\nodes.py
```

```powershell
C:\ComfyUI\.venv\Scripts\python.exe -c "import importlib.util; p=r'C:\ComfyUI\custom_nodes\ComfyUI-SAM3DBody-openpose-editor\nodes.py'; s=importlib.util.spec_from_file_location('n', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(sorted(m.NODE_CLASS_MAPPINGS.keys()))"
```

最後の状態ではこの2つは通っていた。

## 最終状態の保存ファイル

この設計書と照合する実体は次。

- `nodes_failed_sam31_attempt_20260724.py`

これは最後の状態の退避版であり、会話ログから再現する場合の最も近い参照実装である。




