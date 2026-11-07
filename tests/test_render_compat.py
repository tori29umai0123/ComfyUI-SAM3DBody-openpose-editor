import sys
import unittest

from pathlib import Path

import nodes

sys.path.append(str(Path(__file__).resolve().parents[3]))


def pose(arm_length=10):
    body = [0.0, 0.0, 1.0] * 18
    for index, x, y in ((1, 50, 30), (2, 60, 30), (3, 60 + arm_length, 30),
                        (8, 45, 60), (11, 55, 60), (9, 45, 80), (12, 55, 80)):
        body[3 * index:3 * index + 3] = [x, y, 1.0]
    return [{"canvas_width": 100, "canvas_height": 100,
             "people": [{"pose_keypoints_2d": body, "face_keypoints_2d": [],
                         "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": []}]}]


class RenderCompatibilityTest(unittest.TestCase):
    def test_bundled_renderer_without_reference(self):
        util = nodes._openpose_util()
        self.assertEqual(Path(util.__file__).resolve(), Path(nodes.OPENPOSE_UTIL_PATH).resolve())
        image, = nodes.SAM3DBodyOpenPoseRender().render(pose())
        self.assertEqual(tuple(image.shape), (1, 100, 100, 3))

    def test_bundled_renderer_with_reference(self):
        original, = nodes.SAM3DBodyOpenPoseRender().render(pose(10))
        image, = nodes.SAM3DBodyOpenPoseRender().render(pose(10), body_shape_keypoint=pose(20))
        self.assertEqual(tuple(image.shape), (1, 100, 100, 3))
        self.assertTrue((original != image).any())


if __name__ == "__main__":
    unittest.main()
