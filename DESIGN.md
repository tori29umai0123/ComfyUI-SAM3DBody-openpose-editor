# SAM3D Body OpenPose Analyze / Render 設計書

## 目的

ComfyUI 上で SAM3D Body の人物推定結果を OpenPose 互換の `POSE_KEYPOINT` として出力し、OpenPose Editor 系ノードで扱えるようにする。

このノードは、まず「壊れずに人物ポーズを出す」ことを最優先にする。AnimeHeads、anime-face-detector、RT-DETR、SAM3.1 連携は追加機能として段階的に入れる。最小版が安定するまで、複数の検出器を同時に絡めない。

## ノード構成

### SAM3D Body OpenPose Analyze

入力:

- `pose_image`: 解析対象画像
- `body_shape_image`: 任意。体型参照用画像
- `device`: `cuda` / `cpu`
- `bbox_threshold`
- `inference_type`: `full` / `body` / `hand`
- `confidence`
- `auto_head_proportion`
- `head_confidence`

出力:

- `pose_keypoint`: OpenPose互換の人物ポーズ
- `body_shape_keypoint`: 体型参照用の OpenPose + SAM3DBody metadata
- `debug_person_crops`: デバッグ用画像。最小版では空画像でよい

UIに出さないもの:

- `model_path`

SAM3D Body model path は `ComfyUI/models/sam3dbody` 決め打ち。存在しない場合は自動DLする。

### SAM3D Body OpenPose Render

入力:

- `pose_keypoint`
- 任意 `body_shape_keypoint`
- 表示制御: body / face / hands
- サイズ、offset、zoom
- 各部位スケール

出力:

- `openpose_image`

OpenPose描画は `ComfyUI-ultimate-openpose-editor/util.py` の `draw_pose_json` を使う。

## 最小復旧版の仕様

最初に作るべき安定版は以下だけを満たす。

1. SAM3D Body model を固定パスからロードする
2. 画像1枚を SAM3DBodyEstimator に渡す
3. `pred_keypoints_2d` を COCO-18 + OpenPose手形式へ変換する
4. `POSE_KEYPOINT` を返す
5. Render で描画できる
6. `model_path` UIを出さない
7. 複数人は SAM3DBody が返した bbox を top-left 順に並べる
8. 失敗時は分かるエラーを出す

最小復旧版では、以下を入れない。

- AnimeHeads
- anime-face-detector
- RT-DETR body bbox
- SAM3.1 mask
- 後ろ向き/横向き顔パーツ削除
- 顔ランドマーク独自描画

理由: これらはすべて人物対応・座標変換・可視判定が絡むため、最小動作確認前に入れると原因切り分けが難しくなる。

## 段階的な追加機能

### Phase 1: AnimeHeads による人数・頭bbox

目的:

- 集合絵で人物数を head 基準に安定させる
- 顔が見えていない後頭部も head として数える

実装:

- `models/animeheads/animeheadsv3.onnx` を自動DL
- `head_id` を付ける
- head bbox を top-left 順で保持

出力metadata例:

```json
{
  "sam3dbody_guidance": [
    {
      "head_id": 0,
      "head_bbox": [x1, y1, x2, y2],
      "head_confidence": 0.9
    }
  ]
}
```

### Phase 2: RT-DETRv4 body bbox

目的:

- head ごとに body bbox を割り当て、SAM3DBody の人物cropを安定させる

使用モデル:

- `tori29umai/rtdetrv4-x-manga109s_v2`
- 保存先: `models/rtdetrv4/rtdetrv4-x-manga109s_v2.onnx`

実装:

- RT-DETR body bbox を検出
- head 中心と body bbox の包含/距離でマッチング
- head に対応する body bbox がない場合は head から推定bboxを作る
- SAM3DBody には `bboxes=` として渡す

重要:

- この段階では mask は作らない
- `head_id -> body_bbox` の対応表を出力metadataに保持する

### Phase 3: anime-face-detector 顔ランドマーク

目的:

- OpenPose式の鼻・目・耳とは別に、アニメ顔の詳細keypointを保持/描画する

実装:

- まず全体画像で通常検出
- 小さい顔は crop を2倍程度に拡大して検出
- AnimeHeads head bbox を detector の `boxes=` に渡す誘導検出を追加
- ただし誘導検出だけの顔は、通常顔検出と近い場合のみ採用する

注意:

- 後頭部の head bbox から顔ランドマークを捏造しない
- 横顔で通常検出が弱い場合、消えすぎる可能性があるので閾値はログで調整する

### Phase 4: OpenPose 顔点の可視判定

対象点:

- `0`: nose
- `14`: right eye
- `15`: left eye
- `16`: right ear
- `17`: left ear

目的:

- 後ろ向き、横向きで見えていない鼻・目・耳を非表示にする

基本方針:

- OpenPose body の顔点は SAM3DBody 由来なので、顔が見えなくても出ることがある
- 表示するかどうかは anime-face-detector の通常顔検出を基準にする
- AnimeHeads誘導だけの顔は「可視顔」として扱わない

判定:

- head bbox は存在するが通常顔検出がない: 顔点を消す
- 通常顔検出bbox内にある鼻/目/耳点: 残す
- 通常顔検出bbox外の鼻/目/耳点: confidence を `0.0` にする

### Phase 5: SAM3.1 mask 連携

これは後回しにする。

目的:

- body bbox を prompt にして人物maskを作る
- そのmask内で顔・手・体の対応判定を行う

未解決点:

- ComfyUI本体の `SAM31` は `sam3.1_multiplex_fp16.safetensors` を認識する
- ただし現在の簡単な `Sam3Processor` API は `sam3.pt` 形式前提
- `.safetensors` の SAM3.1 を bbox prompt で直接使う薄いwrapperが未設計

採用条件:

- ComfyUI本体の SAM31 model loader から bbox prompt mask を取得するAPIを明確にする
- 1枚画像・1bboxで mask が返る最小テストを先に作る
- その後で Analyze に組み込む

禁止:

- SAM3.1が不明なまま RTMDet-Ins mask を「SAM3.1扱い」にしない
- mask 取得に失敗した状態で後段判定を変えない

## データ構造

`pose_keypoint`:

```json
{
  "people": [
    {
      "pose_keypoints_2d": [],
      "face_keypoints_2d": [],
      "hand_left_keypoints_2d": [],
      "hand_right_keypoints_2d": []
    }
  ],
  "canvas_width": 1024,
  "canvas_height": 768,
  "sam3dbody_guidance": []
}
```

`body_shape_keypoint` は上記に加えて `sam3dbody_people` を持つ。

```json
{
  "sam3dbody_people": [
    {
      "format": "mhr70",
      "bbox": [],
      "guidance": {},
      "keypoints_2d": [],
      "keypoints_3d": [],
      "joint_coords": [],
      "joint_rotations": [],
      "camera": [],
      "focal_length": [],
      "pose_params": {}
    }
  ]
}
```

## 実装順序

1. `nodes.py` を最小復旧版にする
2. `py_compile` と import check を通す
3. 実画像1枚で Analyze -> Render が通ることを確認
4. AnimeHeads だけ追加
5. head数とSAM3DBody出力人数が合うか確認
6. RT-DETR body bbox を追加
7. head/body 対応metadataを確認
8. anime-face-detector を追加
9. 顔ランドマーク描画を確認
10. OpenPose 顔点の可視判定を追加
11. SAM3.1 は独立PoCが通ってから追加

## 検証項目

最低限:

- ComfyUI起動時にノード登録が失敗しない
- Analyze UI に `model_path` が出ない
- 正面1人で pose が出る
- 正面複数人で人数が大きく崩れない
- Render が黒画面にならない

追加機能ごと:

- AnimeHeads head bbox 数をログに出す
- RT-DETR body bbox 数をログに出す
- head/body マッチ結果を metadata に出す
- anime-face-detector の通常検出数と誘導検出数をログに出す
- 顔点削除数をログに出す

## ロールバック方針

大きな機能を入れる前に必ず `nodes_YYYYMMDD_phaseX.py` としてコピーを残す。

壊れた場合は、直前phaseのバックアップに戻す。複数機能を一度に入れない。

## 現在の既知バックアップ

- `nodes_rebuild_working_backup.py`: 会話履歴上の最古のまともな復旧版
- `nodes_before_head_body_mask_patch.py`: 顔誘導フィルタ追加後
- `nodes_failed_sam31_attempt_20260724.py`: SAM3.1試行で悪化した版
- `nodes_restored_but_still_bad_20260724.py`: 中間復元版

## 結論

次に作るべきものは、機能全部入りではなく最小復旧版。

最初の成功条件は「SAM3DBody -> OpenPose -> Render が安定して通る」こと。その後、AnimeHeads、RT-DETR、anime-face-detector、顔点可視判定、SAM3.1の順で足す。
