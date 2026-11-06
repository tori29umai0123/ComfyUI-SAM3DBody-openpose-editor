import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

import nodes


class RenderBodyShapeTests(unittest.TestCase):
    def test_each_person_uses_its_matching_body_shape(self):
        pose = [{"canvas_width": 8, "canvas_height": 8,
                 "people": [{"id": "pose1", "face_keypoints_2d": []},
                            {"id": "pose2", "face_keypoints_2d": []}]}]
        shapes = [{"canvas_width": 8, "canvas_height": 8,
                   "people": [{"id": "shape1"}, {"id": "shape2"}]}]
        pairs = []

        def draw(source_json, *_args, target_pose_keypoint_obj=None):
            person = json.loads(source_json)[0]["people"][0]
            target = target_pose_keypoint_obj[0]["people"][0]
            pairs.append((person["id"], target["id"]))
            color = [255, 0, 0] if person["id"] == "pose1" else [0, 255, 0]
            return [np.tile(np.asarray(color, dtype=np.uint8), (8, 8, 1))], None

        with patch.object(nodes, "_openpose_util", return_value=Mock(draw_pose_json=draw)):
            image, = nodes.SAM3DBodyOpenPoseRender().render(pose, body_shape_keypoint=shapes)

        self.assertEqual(pairs, [("pose1", "shape1"), ("pose2", "shape2")])
        np.testing.assert_array_equal(image[0, 0, 0].numpy(), [1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
