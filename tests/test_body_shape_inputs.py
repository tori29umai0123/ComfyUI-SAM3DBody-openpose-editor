import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import nodes


class BodyShapeInputTests(unittest.TestCase):
    def test_append_nodes_preserve_connection_and_batch_order(self):
        first = torch.zeros((1, 8, 8, 3))
        second = torch.ones((2, 8, 8, 3))
        third = torch.full((1, 8, 8, 3), 2.0)
        append = nodes.SAM3DBodyOpenPoseBodyShapeAppend()
        collection, = append.append(second, append.append(first)[0])
        collection, = append.append(third, collection)

        frames = list(nodes._body_shape_frames(body_shape_images=collection))
        self.assertEqual([(source is first, source is second, source is third, index)
                          for source, index in frames],
                         [(True, False, False, 0), (False, True, False, 0),
                          (False, True, False, 1), (False, False, True, 0)])

    def test_single_reference_precedes_collection(self):
        single = torch.zeros((1, 8, 8, 3))
        multiple = torch.ones((1, 8, 8, 3))
        collection, = nodes.SAM3DBodyOpenPoseBodyShapeAppend().append(multiple)
        frames = list(nodes._body_shape_frames(single, collection))
        self.assertIs(frames[0][0], single)
        self.assertIs(frames[1][0], multiple)

    def test_missing_reference_keeps_its_position_and_falls_back(self):
        pose_outputs = [object(), object(), object()]
        pose_people = [object(), object(), object()]
        shape_output, shape_face = object(), object()
        outputs, faces = nodes._body_shape_slots_for_pose(
            pose_outputs, pose_people, [(None, None), (shape_output, shape_face)])
        self.assertIs(outputs[0], pose_outputs[0])
        self.assertIs(outputs[1], shape_output)
        self.assertIs(outputs[2], pose_outputs[2])
        self.assertIs(faces[0], pose_people[0])
        self.assertIs(faces[1], shape_face)
        self.assertIs(faces[2], pose_people[2])

    def test_new_socket_type_matches_append_output(self):
        append_type = nodes.SAM3DBodyOpenPoseBodyShapeAppend.RETURN_TYPES[0]
        analyze_type = nodes.SAM3DBodyOpenPoseAnalyze.INPUT_TYPES()["optional"]["body_shape_images"][0]
        self.assertEqual(append_type, analyze_type)

    def test_analyze_keeps_later_reference_at_its_person_index(self):
        pose_image = torch.zeros((1, 8, 8, 3))
        missing = torch.ones((1, 8, 8, 3))
        second = torch.full((1, 8, 8, 3), 2.0)

        def output(marker):
            return {"bbox": [marker, 0, marker + 1, 1],
                    "pred_keypoints_2d": np.full((70, 2), marker, dtype=np.float32)}

        pose_outputs = [output(10), output(20), output(30)]

        def estimate(_estimator, image, *_args):
            if image is pose_image:
                return pose_outputs, 8, 8, []
            if image is missing:
                raise RuntimeError("missing reference")
            if image is second:
                return [output(99)], 8, 8, []
            raise AssertionError("unexpected image")

        fake_body = SimpleNamespace(SAM3DBodyEstimator=lambda *_args, **_kwargs: object())
        with patch.dict(sys.modules, {"sam_3d_body": fake_body}), \
             patch.object(nodes, "_make_model_config", return_value={}), \
             patch.object(nodes, "_load_sam3d_model", return_value={"model": None, "model_cfg": None}), \
             patch.object(nodes, "_run_sam3d_estimator", side_effect=estimate):
            _, body_shape, _ = nodes.SAM3DBodyOpenPoseAnalyze().analyze(
                pose_image, auto_head_proportion=False, body_shape_images=(missing, second))

        markers = [person["keypoints_2d"][0][0] for person in body_shape[0]["sam3dbody_people"]]
        self.assertEqual(markers, [10.0, 99.0, 30.0])


if __name__ == "__main__":
    unittest.main()
