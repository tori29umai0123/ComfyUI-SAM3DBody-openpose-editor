"""Install packages required by SAM3DBody."""

import platform
import subprocess
import sys


def main():
    requirements = [
        "comfy-env==0.1.75", "comfy-3d-viewers==0.2.33",
        "numpy>=1.21.0,<2.0.0", "pillow>=9.0.0",
        "opencv-python-headless>=4.8.0", "huggingface-hub>=0.16.0",
        "pytorch-lightning>=2.0.0", "omegaconf>=2.3.0",
        "yacs>=0.1.8", "einops>=0.6.0", "timm>=0.9.0",
        "roma>=1.3.0", "braceexpand>=0.1.7",
    ]
    if platform.system().lower() == "windows":
        requirements.append("msvc-runtime")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])


if __name__ == "__main__":
    main()
