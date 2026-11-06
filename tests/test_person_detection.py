import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import nodes
import sam31_face


class PersonDetectionTests(unittest.TestCase):
    def test_sam31_boxes_and_masks_are_passed_to_body_estimator(self):
        estimator = Mock()
        estimator.process_one_image.return_value = [{"bbox": [10, 20, 90, 95]}]
        person = {"box": [10, 20, 90, 95], "mask": np.ones((100, 100), dtype=bool), "score": .9}
        image = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
        with patch.object(sam31_face, "detect_people", return_value=[person]) as detect:
            outputs, width, height, people = nodes._run_sam3d_estimator(estimator, image, .8, "body")
        detect.assert_called_once()
        self.assertEqual((width, height), (100, 100))
        self.assertEqual(people, [person])
        self.assertEqual(len(outputs), 1)
        arguments = estimator.process_one_image.call_args.kwargs
        np.testing.assert_array_equal(arguments["bboxes"], [[10, 20, 90, 95]])
        np.testing.assert_array_equal(arguments["masks"], np.ones((1, 100, 100), dtype=np.uint8))
        self.assertEqual(arguments["inference_type"], "body")

    def test_no_sam31_people_is_reported(self):
        estimator = Mock()
        with patch.object(sam31_face, "detect_people", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "SAM3.1 did not detect"):
                nodes._run_sam3d_estimator(estimator, torch.zeros((1, 32, 32, 3)), .8, "body")
        estimator.process_one_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
