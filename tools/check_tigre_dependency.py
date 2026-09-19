from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    report = {
        "schema": "x2cbct_tigre_dependency_check.v1",
        "python": sys.executable,
        "nvcc": shutil.which("nvcc"),
        "cl": shutil.which("cl"),
        "tigre_import": False,
        "tigre_error": None,
    }
    try:
        import tigre  # type: ignore

        report["tigre_import"] = True
        report["tigre_path"] = getattr(tigre, "__file__", None)
    except Exception as exc:
        report["tigre_error"] = repr(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["tigre_import"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
