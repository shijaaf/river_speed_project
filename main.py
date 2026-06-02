"""
Main entry point for the River Speed Estimation Project.
This file is used to test whether the project environment works correctly.
"""

import sys
import cv2
import numpy as np
import pandas as pd
import sklearn
import torch


def main():
    """
    Print installed package versions to verify the environment setup.
    """

    print("River Speed Estimation Project")
    print("--------------------------------")

    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. CPU will be used.")


if __name__ == "__main__":
    main()