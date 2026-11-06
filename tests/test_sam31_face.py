import unittest
import os
import tempfile
from unittest.mock import patch

import numpy as np
import sam31_face
import torch
from sam31_face import update_pose


def point(x, y, score=.9):
    return {"x": x, "y": y, "score": score, "box": [x-3, y-3, x+3, y+3]}


def person():
    body = [0.0, 0.0, .9] * 18
    for idx, x, y in ((0, 50, 55), (14, 40, 45), (15, 60, 45), (16, 25, 48), (17, 75, 48)):
        body[idx*3:idx*3+3] = [x, y, .9]
    return {"pose_keypoints_2d": body}


class Sam31FacePoseTests(unittest.TestCase):
    def test_rear_head_hides_facial_bones(self):
        pose = update_pose({"people": [person()]}, [{"ear": [point(25, 48)]}])
        body = pose["people"][0]["pose_keypoints_2d"]
        self.assertEqual([body[i*3+2] for i in (0, 14, 15, 16, 17)], [0]*5)
        self.assertEqual(pose["people"][0]["face_keypoints_2d"], [0.0] * 210)

    def test_side_face_uses_visible_eye_and_nose(self):
        pose = update_pose({"people": [person()]}, [{"eye": [point(39, 44)], "nose": [point(51, 56)]}])
        body = pose["people"][0]["pose_keypoints_2d"]
        self.assertEqual(body[14*3:14*3+2], [39.0, 44.0])
        self.assertEqual(body[0:2], [51.0, 56.0])
        self.assertEqual(body[15*3+2], 0)
        face = pose["people"][0]["face_keypoints_2d"]
        self.assertEqual(len(face), 210)
        self.assertGreater(face[36*3+2], 0)
        self.assertEqual(face[42*3+2], 0)
        self.assertEqual(face[0*3+2], 0)

    def test_hair_and_head_accessories_are_not_ears(self):
        for eyes, mouth, false_ears in (
            ([(330, 390), (552, 387)], (460, 553), [(469, 251), (306, 296)]),
            ([(602, 467), (858, 436)], (740, 647), [(1093, 99), (507, 94)]),
        ):
            parts = {"eye": [point(*xy) for xy in eyes], "mouth": [point(*mouth)],
                     "ear": [point(*xy) for xy in false_ears]}
            result = update_pose({"people": [person()]}, [parts])
            body = result["people"][0]["pose_keypoints_2d"]
            self.assertEqual([body[index * 3 + 2] for index in (16, 17)], [.25, .25])
            for index in (16, 17):
                inferred = np.asarray(body[index * 3:index * 3 + 2])
                self.assertGreater(min(np.linalg.norm(inferred - np.asarray(p)) for p in false_ears), 45)

    def test_visible_lateral_ear_is_retained(self):
        parts = {"eye": [point(40, 45), point(60, 45)], "mouth": [point(50, 70)],
                 "ear": [point(25, 53), point(75, 53)]}
        result = update_pose({"people": [person()]}, [parts])
        body = result["people"][0]["pose_keypoints_2d"]
        self.assertGreater(body[16 * 3 + 2], 0)
        self.assertGreater(body[17 * 3 + 2], 0)
        self.assertEqual(body[16 * 3:16 * 3 + 2], [25.0, 53.0])
        self.assertEqual(body[17 * 3:17 * 3 + 2], [75.0, 53.0])

    def test_hidden_ears_follow_face_tilt(self):
        parts = {"eye": [point(40, 60), point(60, 40)], "mouth": [point(70, 70)]}
        result = update_pose({"people": [person()]}, [parts])
        body = result["people"][0]["pose_keypoints_2d"]
        left = np.asarray(body[16 * 3:16 * 3 + 2])
        right = np.asarray(body[17 * 3:17 * 3 + 2])
        self.assertEqual(body[16 * 3 + 2], .25)
        self.assertEqual(body[17 * 3 + 2], .25)
        self.assertAlmostEqual(right[0] - left[0], -(right[1] - left[1]), places=5)

    def test_face_outline_expands_hidden_ear_position(self):
        eyes = [point(232, 556), point(455, 526)]
        face = {"x": 358, "y": 570, "score": .9, "box": [124, 282, 593, 859],
                "outline": [[126, 550], [132, 445], [166, 435], [382, 387], [579, 451],
                            [579, 552], [587, 623], [590, 664], [576, 776], [468, 842],
                            [388, 858], [229, 769], [175, 642], [133, 582]]}
        result = update_pose({"people": [person()]}, [{"face": [face], "eye": eyes,
                                                     "mouth": [point(396, 741)]}])
        body = result["people"][0]["pose_keypoints_2d"]
        self.assertGreater(body[17 * 3], 585)
        self.assertLess(body[16 * 3], 140)

    def test_mouth_misdetected_as_nose_is_ignored(self):
        parts = {"eye": [point(40, 45), point(60, 45)],
                 "mouth": [point(50, 70)], "nose": [point(50, 70)]}
        result = update_pose({"people": [person()]}, [parts])
        body = result["people"][0]["pose_keypoints_2d"]
        self.assertEqual(body[2], .25)
        self.assertGreater(body[1], 45)
        self.assertLess(body[1], 70)

    def test_detected_parts_fill_openpose_70_slots(self):
        face_box = {"x": 50, "y": 50, "score": .9, "box": [20, 20, 80, 90]}
        parts = {"face": [face_box], "eye": [point(40, 45), point(60, 45)],
                 "eyebrow": [point(40, 35), point(60, 35)],
                 "nose": [point(50, 55)], "mouth": [point(50, 70)]}
        pose = update_pose({"people": [person()]}, [parts])
        landmarks = pose["people"][0]["face_keypoints_2d"]
        self.assertEqual(len(landmarks), 70 * 3)
        self.assertTrue(all(landmarks[index * 3 + 2] > 0 for index in range(70)))
        jaw_x = [landmarks[index * 3] for index in range(17)]
        jaw_y = [landmarks[index * 3 + 1] for index in range(17)]
        self.assertLess(jaw_x[1] - jaw_x[0], jaw_x[8] - jaw_x[7])
        self.assertGreater(jaw_y[8], jaw_y[4])
        self.assertAlmostEqual(jaw_y[4], jaw_y[12])
        self.assertEqual(len(pose["face_keypoints"]["faces"][0]["keypoints"]), 70)

    def test_eyebrow_points_follow_detected_mask_instead_of_box(self):
        def brow(x):
            outline = [[x, 42], [x + 10, 34], [x + 20, 30], [x + 30, 34], [x + 40, 42],
                       [x + 40, 46], [x + 30, 38], [x + 20, 34], [x + 10, 38], [x, 46]]
            return {"x": x + 20, "y": 38, "score": .9, "box": [x, 10, x + 40, 70],
                    "outline": outline}
        eyes = [{"x": x, "y": 55, "score": .9, "box": [x-15, 45, x+15, 65]}
                for x in (40, 80)]
        parts = {"eye": eyes, "eyebrow": [brow(20), brow(60)]}
        result = update_pose({"people": [person()]}, [parts])
        self.assertFalse(result["sam31_face_parts"]["faces"][0]["parts"]["eyebrow"][0].get("inferred", False))
        face = result["people"][0]["face_keypoints_2d"]
        self.assertGreater(face[19 * 3 + 1], 30)
        self.assertLess(face[19 * 3 + 1], 40)
        self.assertGreater(face[24 * 3 + 1], 30)

    def test_eyelashes_are_not_used_as_eyebrows(self):
        eyes = [{"x": 232, "y": 556, "score": .89, "box": [203, 519, 260, 594]},
                {"x": 455, "y": 526, "score": .93, "box": [422, 486, 487, 566]}]
        lashes = [{"x": 203, "y": 516, "score": .85, "box": [140, 474, 266, 558]},
                  {"x": 480, "y": 465, "score": .92, "box": [389, 428, 570, 502]}]
        hair = [{"x": 180, "y": 411, "score": .5, "box": [119, 376, 241, 447]},
                {"x": 435, "y": 378, "score": .58, "box": [329, 364, 542, 392]}]
        brows = sam31_face._select_or_infer_brows({"eye": eyes, "eyebrow": lashes + hair,
                                                   "mouth": [point(396, 741)]})
        self.assertEqual(len(brows), 2)
        self.assertTrue(all(brow.get("inferred") for brow in brows))
        self.assertTrue(445 < brows[0]["y"] < 480)
        self.assertTrue(415 < brows[1]["y"] < 450)
        for brow in brows:
            self.assertEqual(len(brow["landmarks"]), 5)

    def test_separate_eyebrow_detection_is_kept(self):
        eyes = [point(40, 60), point(80, 60)]
        brow = {"x": 40, "y": 40, "score": .8, "box": [34, 35, 46, 45]}
        brows = sam31_face._select_or_infer_brows({"eye": eyes, "eyebrow": [brow],
                                                   "mouth": [point(60, 90)]})
        self.assertIs(brows[0], brow)
        self.assertTrue(brows[1].get("inferred"))

    def test_closed_eyes_and_mouth_do_not_draw_open_interiors(self):
        closed_left = {"x": 40, "y": 45, "score": .9, "box": [32, 43, 48, 46]}
        closed_right = {"x": 60, "y": 45, "score": .9, "box": [52, 43, 68, 46]}
        closed_mouth = {"x": 50, "y": 70, "score": .9, "box": [40, 69, 60, 72]}
        parts = {"face": [{"x": 50, "y": 55, "score": .9, "box": [20, 20, 80, 90]}],
                 "closed_eye": [closed_left, closed_right], "closed_mouth": [closed_mouth],
                 "nose": [point(50, 55)]}
        pose = update_pose({"people": [person()]}, [parts])
        face = pose["people"][0]["face_keypoints_2d"]
        self.assertGreater(face[36 * 3 + 2], 0)
        self.assertEqual(face[68 * 3 + 2], 0)
        self.assertEqual(face[69 * 3 + 2], 0)
        self.assertEqual(face[60 * 3 + 2], 0)
        self.assertGreater(face[48 * 3 + 2], 0)

    def test_rotated_mask_controls_eye_closure_and_point_tilt(self):
        angle = np.deg2rad(-35)
        along = np.array([np.cos(angle), np.sin(angle)])
        across = np.array([-along[1], along[0]])
        def eye_at(center):
            outline = [(np.array(center) + along * (18 * np.cos(t)) + across * (4 * np.sin(t))).tolist()
                       for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)]
            coords = np.asarray(outline)
            return {"x": center[0], "y": center[1], "score": .9,
                    "box": [*coords.min(axis=0), *coords.max(axis=0)], "outline": outline}
        left, right = eye_at((35, 55)), eye_at((65, 35))
        self.assertTrue(sam31_face._is_closed_feature(left, .38))
        pose = update_pose({"people": [person()]}, [{"eye": [left, right]}])
        face = pose["people"][0]["face_keypoints_2d"]
        self.assertEqual(face[68 * 3 + 2], 0)
        self.assertEqual(face[69 * 3 + 2], 0)
        self.assertLess(face[39 * 3 + 1], face[36 * 3 + 1])
        for index in range(36, 48):
            p = np.asarray(face[index * 3:index * 3 + 2])
            outline = np.asarray((left if index < 42 else right)["outline"])
            self.assertLess(np.linalg.norm(outline - p, axis=1).min(), 4.5)

    def test_face_mask_outline_follows_tilt(self):
        angle = np.deg2rad(30)
        horizontal = np.array([np.cos(angle), np.sin(angle)])
        vertical = np.array([-np.sin(angle), np.cos(angle)])
        center = np.array([50.0, 60.0])
        outline = [(center + horizontal * (35 * np.cos(t)) + vertical * (38 * np.sin(t))).tolist()
                   for t in np.linspace(0, 2 * np.pi, 120, endpoint=False)]
        face = {"outline": outline}
        left = {"x": 50 - 15 * horizontal[0], "y": 45 - 15 * horizontal[1]}
        right = {"x": 50 + 15 * horizontal[0], "y": 45 + 15 * horizontal[1]}
        jaw = sam31_face._jaw_from_outline(face, left, right, {"x": 42, "y": 75})
        self.assertEqual(len(jaw), 17)
        self.assertGreater(jaw[-1][1] - jaw[0][1], 10)
        self.assertGreater(jaw[8][1], 70)

    def test_face_mask_is_extracted_with_detection(self):
        mask = torch.full((1, 1, 100, 100), -1.0)
        yy, xx = torch.meshgrid(torch.arange(100), torch.arange(100), indexing="ij")
        mask[0, 0][((xx - 50) ** 2 / 35 ** 2 + (yy - 50) ** 2 / 40 ** 2) < 1] = 1.0
        result = {"scores": torch.tensor([[3.0]]),
                  "boxes": torch.tensor([[[15.0, 10.0, 85.0, 90.0]]]), "masks": mask}
        candidates = sam31_face._part_candidates(result, (0, 0, 100, 100), .5, .95, True)
        self.assertEqual(len(candidates), 1)
        self.assertGreater(len(candidates[0]["outline"]), 12)

    def test_eyepatch_keeps_hidden_bone_with_low_confidence(self):
        patch = {"x": 60, "y": 45, "score": .9, "box": [53, 37, 68, 54]}
        pose = update_pose({"people": [person()]}, [{"eye": [point(40, 45)], "nose": [point(50, 55)], "eyepatch": [patch]}])
        body = pose["people"][0]["pose_keypoints_2d"]
        self.assertEqual(body[14*3+2], .9)
        self.assertEqual(body[15*3:15*3+3], [60.0, 45.0, .25])

    def test_body_shape_keeps_measured_head_size(self):
        source = person()
        target = person()
        target["pose_keypoints_2d"][14*3:14*3+3] = [47.0, 48.0, .9]
        target["pose_keypoints_2d"][2*3:2*3+3] = [20.0, 80.0, .9]
        shape = {"people": [target], "sam3dbody_people": [{"format": "mhr70"}]}
        result = sam31_face.match_body_shape_head_to_pose(shape, {"people": [source]})
        self.assertEqual(result["people"][0]["pose_keypoints_2d"][14*3:14*3+3], source["pose_keypoints_2d"][14*3:14*3+3])
        self.assertEqual(result["people"][0]["pose_keypoints_2d"][2*3:2*3+3], [20.0, 80.0, .9])
        self.assertEqual(target["pose_keypoints_2d"][14*3:14*3+3], [47.0, 48.0, .9])
        self.assertEqual(result["sam3dbody_people"], shape["sam3dbody_people"])

    def test_face_proportions_follow_body_shape_reference(self):
        def with_face(left, right, brow_y, chin_y):
            p = person()
            body = p["pose_keypoints_2d"]
            for index, x, y in ((1, 50, 100), (8, 40, 200), (11, 60, 200)):
                body[index * 3:index * 3 + 3] = [x, y, .9]
            face = [0.0] * 210
            for index, x, y in ((0, left, 55), (8, 50, chin_y), (16, right, 55),
                                (17, 50, brow_y), (30, 50, 50), (36, left + 8, 45)):
                face[index * 3:index * 3 + 3] = [x, y, .9]
            p["face_keypoints_2d"] = face
            return p

        source = with_face(30, 70, 35, 80)
        target = with_face(20, 80, 30, 90)
        adapted = sam31_face.adapt_pose_face_to_body_shape({"people": [source]}, [target])["people"][0]
        self.assertAlmostEqual(adapted["face_keypoints_2d"][0], 20.0)
        self.assertAlmostEqual(adapted["face_keypoints_2d"][16 * 3], 80.0)
        self.assertEqual(adapted["face_keypoints_2d"][30 * 3:30 * 3 + 2], [50, 50])
        self.assertEqual(adapted["face_keypoints_2d"][60 * 3:60 * 3 + 3], [0.0, 0.0, 0.0])
        self.assertEqual(source["face_keypoints_2d"][0], 30)

    def test_missing_reference_face_keeps_pose_face(self):
        source = person()
        source["face_keypoints_2d"] = [0.0] * 210
        pose = {"people": [source]}
        self.assertIs(sam31_face.adapt_pose_face_to_body_shape(pose, [person()]), pose)

    def test_body_shape_face_scaling_uses_tilted_face_axes(self):
        angle = np.deg2rad(40)
        along = np.array([np.cos(angle), np.sin(angle)])
        down = np.array([-along[1], along[0]])
        def make_person(width):
            p = person()
            body = p["pose_keypoints_2d"]
            for index, xy in ((1, (50, 100)), (8, (40, 200)), (11, (60, 200))):
                body[index * 3:index * 3 + 3] = [*xy, .9]
            for index, xy in ((14, np.array([50, 50]) - along * width / 4),
                              (15, np.array([50, 50]) + along * width / 4)):
                body[index * 3:index * 3 + 3] = [*xy, .9]
            face = [0.0] * 210
            for index, xy in ((0, np.array([50, 60]) - along * width / 2),
                              (16, np.array([50, 60]) + along * width / 2),
                              (8, np.array([50, 60]) + down * 30),
                              (17, np.array([50, 60]) - down * 15),
                              (30, np.array([50, 60]))):
                face[index * 3:index * 3 + 3] = [*xy, .9]
            p["face_keypoints_2d"] = face
            return p
        source, target = make_person(40), make_person(60)
        result = sam31_face.adapt_pose_face_to_body_shape({"people": [source]}, [target])
        face = result["people"][0]["face_keypoints_2d"]
        span = np.asarray(face[16 * 3:16 * 3 + 2]) - face[0:2]
        self.assertAlmostEqual(np.linalg.norm(span), 60, delta=1)
        self.assertAlmostEqual(np.dot(span / np.linalg.norm(span), along), 1, places=4)

    def test_eye_and_mouth_measurement_works_without_jaw(self):
        source = person()
        target = person()
        for p, left, right in ((source, 40, 60), (target, 35, 65)):
            body = p["pose_keypoints_2d"]
            for index, x, y in ((1, 50, 100), (8, 40, 200), (11, 60, 200)):
                body[index * 3:index * 3 + 3] = [x, y, .9]
            face = [0.0] * 210
            for index, x, y in ((68, left, 40), (69, right, 40), (51, 50, 70), (30, 50, 50)):
                face[index * 3:index * 3 + 3] = [x, y, .9]
            p["face_keypoints_2d"] = face
        adapted = sam31_face.adapt_pose_face_to_body_shape({"people": [source]}, [target])
        self.assertAlmostEqual(adapted["people"][0]["face_keypoints_2d"][68 * 3], 35.0)


class Sam31CheckpointTests(unittest.TestCase):
    def test_existing_checkpoint_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "sam3.1", "sam3.1_multiplex_fp16.safetensors")
            os.makedirs(os.path.dirname(checkpoint))
            with open(checkpoint, "wb") as output:
                output.write(b"existing")
            with patch.object(sam31_face, "_CHECKPOINT", checkpoint), patch("model_download.download") as download:
                self.assertEqual(sam31_face.ensure_checkpoint(), checkpoint)
                download.assert_not_called()

    def test_missing_checkpoint_is_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "sam3.1", "sam3.1_multiplex_fp16.safetensors")
            source = os.path.join(directory, "cached.safetensors")
            with open(source, "wb") as output:
                output.write(b"downloaded")
            with patch.object(sam31_face, "_CHECKPOINT", checkpoint), patch("model_download.download", return_value=source) as download:
                self.assertEqual(sam31_face.ensure_checkpoint(), checkpoint)
                download.assert_called_once_with("file", repo_id="Comfy-Org/sam3.1", filename="checkpoints/sam3.1_multiplex_fp16.safetensors")
            with open(checkpoint, "rb") as result:
                self.assertEqual(result.read(), b"downloaded")


if __name__ == "__main__":
    unittest.main()
