from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "external" / "deepsparse_official" / "logs" / "pretrain" / "ep_700.pth"
EXPECTED_SHA256 = "36c5aa0b1f804702668c757a8da8ecf17fb6f6ebab8c9e977da57a0fa1656d59"


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"missing DeepSparse foundation checkpoint: {CHECKPOINT}")
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {digest} != {EXPECTED_SHA256}")
    state = torch.load(CHECKPOINT, map_location="cpu")
    if not isinstance(state, dict) or len(state) != 210:
        raise RuntimeError("unexpected DeepSparse foundation checkpoint structure")
    report = {
        "status": "pass",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": digest,
        "checkpoint_tensor_count": len(state),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
