import unittest
from unittest.mock import Mock, patch

import numpy as np
import nodes


class PersonDetectionTests(unittest.TestCase):
    def test_rtdetrv4_receives_model_inputs_and_original_size(self):
        session = Mock()
        session.run.return_value = (
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([[[10, 20, 100, 90], [120, 10, 180, 90]]], dtype=np.float32),
            np.asarray([[.9, .8]], dtype=np.float32),
        )
        with patch.object(nodes, "_load_rtdetrv4_body_session", return_value=session):
            boxes = nodes._detect_rtdetrv4_body_persons(np.zeros((100, 200, 3), dtype=np.uint8))
        feed = session.run.call_args.args[1]
        self.assertEqual(set(feed), {"images", "orig_target_sizes"})
        self.assertEqual(feed["images"].shape, (1, 3, 1280, 1280))
        self.assertEqual(feed["images"].dtype, np.float32)
        np.testing.assert_array_equal(feed["orig_target_sizes"], [[200, 100]])
        self.assertEqual([item["bbox"] for item in boxes], [[10.0, 20.0, 100.0, 90.0]])


if __name__ == "__main__":
    unittest.main()
