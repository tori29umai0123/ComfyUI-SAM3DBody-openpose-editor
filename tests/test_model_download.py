import unittest
import json
from unittest.mock import patch

import model_download


class ModelDownloadTests(unittest.TestCase):
    def test_normal_download_uses_existing_hub_client(self):
        with patch("huggingface_hub.hf_hub_download", return_value="cached.onnx") as hub, \
                patch("model_download.subprocess.run") as process:
            self.assertEqual(model_download.download("file", repo_id="example/model", filename="model.onnx"), "cached.onnx")
            hub.assert_called_once_with(repo_id="example/model", filename="model.onnx", etag_timeout=30)
            process.assert_not_called()

    def test_only_winerror_10054_triggers_ipv4_retry(self):
        reset = OSError("[WinError 10054] Connection reset by peer")
        completed = type("Result", (), {"returncode": 0,
                                        "stdout": "SAM3DBODY_DOWNLOAD_RESULT:C:\\cached\\model.onnx\n",
                                        "stderr": ""})()
        with patch.object(model_download.sys, "platform", "win32"), \
                patch("huggingface_hub.hf_hub_download", side_effect=reset), \
                patch("model_download.subprocess.run", return_value=completed) as process:
            self.assertEqual(model_download.download("file", repo_id="example/model", filename="model.onnx"),
                             "C:\\cached\\model.onnx")
            self.assertEqual(process.call_count, 1)
            self.assertEqual(process.call_args.args[0][1:3], [model_download.__file__, "--ipv4"])

        with patch.object(model_download.sys, "platform", "win32"), \
                patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("HTTP 403")), \
                patch("model_download.subprocess.run") as process:
            with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
                model_download.download("file", repo_id="example/model", filename="model.onnx")
            process.assert_not_called()

    def test_snapshot_keeps_local_directory_and_serial_transfer(self):
        with patch("huggingface_hub.snapshot_download", return_value="C:\\models\\sam3dbody") as hub:
            self.assertEqual(model_download.download("snapshot", repo_id="example/model", local_dir="C:\\models\\sam3dbody"),
                             "C:\\models\\sam3dbody")
            hub.assert_called_once_with(repo_id="example/model", local_dir="C:\\models\\sam3dbody",
                                        etag_timeout=30, max_workers=1)

    def test_ipv4_worker_configures_local_transport(self):
        payload = json.dumps({"kind": "file", "kwargs": {"repo_id": "example/model", "filename": "model.onnx"}})
        with patch("huggingface_hub.set_client_factory") as set_factory, \
                patch("huggingface_hub.hf_hub_download", return_value="cached.onnx") as hub, \
                patch("httpx.HTTPTransport", return_value=object()) as transport, \
                patch("httpx.Client") as client:
            model_download._run_ipv4(payload)
            factory = set_factory.call_args.args[0]
            factory()
            transport.assert_called_once_with(local_address="0.0.0.0")
            client.assert_called_once()
            self.assertTrue(client.call_args.kwargs["follow_redirects"])
            hub.assert_called_once_with(repo_id="example/model", filename="model.onnx")


if __name__ == "__main__":
    unittest.main()
