# Third-party repositories

This repository does **not** redistribute third-party source code. The baseline
methods and the projector toolbox are consumed as external clones. To reproduce
the paper, clone each repository from its official source and check out the
commit recorded below (the versions used in the paper):

| Directory (working repo) | Project | Pinned commit | Date |
|---|---|---|---|
| `external/TIGRE` | TIGRE CBCT toolbox ([CERN/TIGRE](https://github.com/CERN/TIGRE)) | `3273b4a` | 2024-10-25 |
| `external/dif_net_official` | DIF-Net (MICCAI 2023) | `ddc4c3d` | 2025-08-16 |
| `external/c2rv_cbct_official` | C2RV (CVPR 2024) | `540416f` | 2025-03-10 |
| `external/deepsparse_official` | DeepSparse | `a055aba` | 2026-05-12 |
| `external/dif_gaussian_official` | DIF-Gaussian (MICCAI 2024) | `2c717dc` | 2025-03-09 |
| `external/efficient_capsnet_official` | Efficient-CapsNet | `cd0b533` | 2026-01-10 |
| `external/naf_cbct_official` | NAF-CBCT | `cff7da3` | 2024-10-28 |

Notes:

- Clone each project into `external/<name>` in your working copy so the
  adapters under `src/x2cbct_deepsparse/` and the comparison tools resolve them
  at the documented paths.
- The DeepSparse baseline additionally uses its official finetune config and
  pretrained checkpoint referenced by `configs/deepsparse/base.yaml`; download
  both from the official DeepSparse repository/release.
- Each external project remains under its own license (license files are kept
  inside the respective repositories; TIGRE is BSD-3-Clause). Consult them
  before any redistribution.
- LungMask (R231 weights) used for the downstream segmentation experiments is
  available at the official LungMask repository.
