"""CUDA smoke test for GTX 1070."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        x = torch.randn(4, 4, device="cuda")
        y = x @ x.T
        print(f"CUDA matmul OK: shape={y.shape}")
    else:
        print("WARNING: CUDA not available, will use CPU")
        x = torch.randn(4, 4)
        y = x @ x.T
        print(f"CPU matmul OK: shape={y.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
