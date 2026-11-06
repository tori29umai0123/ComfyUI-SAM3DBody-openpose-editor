import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

HF_WHEEL_REPO = "https://huggingface.co/datasets/tori29umai/anime_face_detector_Wheels/resolve/main"


def run_pip(args):
    cmd = [sys.executable, "-m", "pip", *args]
    print("[SAM3DBodyOpenPose]", " ".join(cmd))
    subprocess.check_call(cmd)


def install_sam3dbody_requirements():
    requirements = [
        "comfy-env==0.1.75",
        "comfy-3d-viewers==0.2.33",
        "numpy>=1.21.0,<2.0.0",
        "pillow>=9.0.0",
        "opencv-python-headless>=4.8.0",
        "huggingface-hub>=0.16.0",
        "pytorch-lightning>=2.0.0",
        "omegaconf>=2.3.0",
        "yacs>=0.1.8",
        "einops>=0.6.0",
        "timm>=0.9.0",
        "roma>=1.3.0",
        "braceexpand>=0.1.7",
        "onnxruntime>=1.17.0",
    ]
    if platform.system().lower() == "windows":
        requirements.append("msvc-runtime")
    run_pip(["install", *requirements])


def install_from_pypi():
    run_pip(["install", "anime-face-detector>=0.1.0"])





def install_hf_wheel(py_minor):
    if py_minor == 11:
        hf_name = "anime_face_detector-0.1.0.post1-python311-py3-none-any.whl"
        pip_name = "anime_face_detector-0.1.0.post1-py3-none-any.whl"
    elif py_minor == 10:
        hf_name = "anime_face_detector-0.1.0.post2-python310-py3-none-any.whl"
        pip_name = "anime_face_detector-0.1.0.post2-py3-none-any.whl"
    else:
        raise RuntimeError(f"Unsupported Hugging Face wheel target: Python 3.{py_minor}")

    url = f"{HF_WHEEL_REPO}/{hf_name}?download=true"
    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded = os.path.join(tmpdir, hf_name)
        pip_wheel = os.path.join(tmpdir, pip_name)
        print(f"[SAM3DBodyOpenPose] Downloading {url}")
        urllib.request.urlretrieve(url, downloaded)
        shutil.copyfile(downloaded, pip_wheel)
        run_pip(["install", pip_wheel])


def patch_pyproject_for_current_python(pyproject_path):
    with open(pyproject_path, "r", encoding="utf-8") as f:
        text = f.read()
    major, minor = sys.version_info[:2]
    if minor < 10:
        raise RuntimeError("anime-face-detector 0.1.x requires Python 3.10 or newer.")
    text = text.replace('requires-python = ">=3.12"', f'requires-python = ">={major}.{minor}"')
    text = text.replace('version = "0.1.0"', f'version = "0.1.0.local{major}{minor}"')
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(text)


def install_self_build():
    url = "https://github.com/hysts/anime-face-detector/archive/refs/heads/main.zip"
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = os.path.join(tmpdir, "anime-face-detector-main.zip")
        print(f"[SAM3DBodyOpenPose] Downloading source {url}")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(tmpdir)
        source_dir = os.path.join(tmpdir, "anime-face-detector-main")
        patch_pyproject_for_current_python(os.path.join(source_dir, "pyproject.toml"))
        run_pip(["install", source_dir])


def main():
    system = platform.system().lower()
    major, minor = sys.version_info[:2]
    if major != 3:
        raise RuntimeError("Python 3 is required.")

    install_sam3dbody_requirements()

    if system in {"windows", "linux"} and minor >= 12:
        print("[SAM3DBodyOpenPose] Installing upstream anime-face-detector from PyPI.")
        install_from_pypi()
    elif system == "windows" and minor in {10, 11}:
        print("[SAM3DBodyOpenPose] Installing anime-face-detector wheel from Hugging Face.")
        install_hf_wheel(minor)
    else:
        print("[SAM3DBodyOpenPose] Installing anime-face-detector from a local patched source build.")
        install_self_build()


if __name__ == "__main__":
    main()
