from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("luna16", "real")
VIEWS = (2, 5, 8, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the six DeepSparse runs into the main-table row")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "output" / "deepsparse",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    values = []
    records = []
    missing = []
    for dataset in DATASETS:
        for view_count in VIEWS:
            path = output_root / f"v{view_count}_{dataset}_noaug" / "metrics_final.json"
            if not path.exists():
                missing.append(str(path))
                continue
            with path.open("r", encoding="utf-8-sig") as handle:
                metrics = json.load(handle)
            mean = metrics["mean"]
            psnr = float(mean["psnr"])
            ssim_percent = float(mean["ssim"]) * 100.0
            values.extend((f"{psnr:.2f}", f"{ssim_percent:.2f}"))
            records.append(
                {
                    "dataset": dataset,
                    "view_count": view_count,
                    "psnr": psnr,
                    "ssim_percent": ssim_percent,
                    "metrics_path": str(path),
                }
            )
    if missing:
        raise FileNotFoundError("missing DeepSparse results:\n" + "\n".join(missing))

    row = "DeepSparse (pretrained) & " + " & ".join(values) + r" \\"
    print(row)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"latex_row": row, "runs": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
