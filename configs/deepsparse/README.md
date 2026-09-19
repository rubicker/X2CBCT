# DeepSparse deployment for X2CBCT

This deployment intentionally uses only the public foundation checkpoint as
initialization.  Official LUNA16/ToothFairy/PANORAMA/PENGWIN downstream weights,
the official random 6--24-view fine-tuning protocol, and Stage 2 are not used.

## Runtime

- Environment: `.conda/deepsparse`
- Python: 3.8.20
- PyTorch: 2.1.2+cu118
- GPU: NVIDIA GeForce RTX 4080
- Official source: `external/deepsparse_official`
- Foundation checkpoint: `external/deepsparse_official/logs/pretrain/ep_700.pth`
- Checkpoint SHA-256: `36c5aa0b1f804702668c757a8da8ecf17fb6f6ebab8c9e977da57a0fa1656d59`

The venv reuses the project's existing CUDA-enabled base environment through
`system-site-packages` and installs DeepSparse-only packages locally.  This
avoids duplicating the CUDA/PyTorch installation while keeping the executable
and extra dependencies separate.

## Protocol

Each YAML file is a complete experiment.  There are three fixed-view models per
dataset (six runs total).  A run sees exactly 2, 5, or 10 project-selected views
during both fine-tuning and testing.  It trains for 200 epochs, samples 10,000
query points per case, and evaluates the final checkpoint at 128^3 using the
project PSNR/SSIM implementation.

The adapter reuses the existing X2CBCT cache, CT/projection normalization,
fixed angles, clinical geometry, and detector-validity masks.  It adds the 32^3
low-resolution back-projection grid expected by DeepSparse.  Per-point validity
is applied before DeepSparse max pooling; all-invalid samples are zero-filled.

## Commands

Environment and checkpoint check:

```powershell
.conda\deepsparse\Scripts\python.exe tools\check_deepsparse_environment.py
```

Recreate the isolated runtime and download/check the foundation checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_deepsparse_environment.ps1
```

One-run smoke test:

```powershell
.conda\deepsparse\Scripts\python.exe tools\train_deepsparse.py `
  --config configs\deepsparse\v2_luna16_noaug.yaml --smoke
```

All six full runs, sequentially:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_deepsparse_all.ps1
```

Preview the exact six commands without starting training:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_deepsparse_all.ps1 -DryRun
```

The order is LUNA16 2/5/10-view followed by real head-and-neck 2/5/10-view.
Each run, including its final evaluation, must finish before the next starts.

After all runs finish, generate the LaTeX row:

```powershell
.conda\deepsparse\Scripts\python.exe tools\collect_deepsparse_table_row.py `
  --json-output runs\output\deepsparse\table_row.json
```
