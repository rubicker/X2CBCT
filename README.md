# CaSE-CBCT: Capsule Self-Attention-Based Extremely Sparse-View CBCT Reconstruction

Official PyTorch implementation of **"CaSE-CBCT: Capsule Self-Attention-Based Extremely Sparse-View CBCT Reconstruction"** (IEEE Transactions on Medical Imaging, under review).

CaSE-CBCT reconstructs diagnostic-quality CBCT volumes from only **5 / 8 / 10 X-ray projections** with an implicit neural representation (INR) pipeline. Its two core components are:

- **DL-CCR** — dual-layer competitive capsule routing that fuses geometry-aligned per-view features and attenuates invalid or low-contribution projections (detector field-of-view truncation, anatomical overlap).
- **Pose-synchronized data augmentation** — a DRR-free augmentation that expands real clinical samples into a multi-pose dataset (S-I translation x yaw rotation) and teaches the model to reproduce patient setup errors for image-guided radiotherapy.

## Repository layout

```
src/
  x2cbct_difnet/       Core method: capsule routing, INR decoder, cone-beam geometry,
                       TIGRE-based DRR generation, setup-error suite (registration,
                       resampling, ROI, manifests), FDK / SART baselines, metrics
  x2cbct_deepsparse/   Adapter for the DeepSparse baseline (external repo required)
  x2cbct_lungmask/     LungMask R231 downstream segmentation experiments
configs/
  deepsparse/          Self-contained run configs (5/8/10-view, NoAug/Aug) + runtime requirements
  lungmask/            Segmentation experiment configs
tools/                 Dataset construction, training, evaluation, analysis entry points
tests/                 Unit tests (synthetic fixtures; no data required, run with pytest)
data_manifest/         Split registries for public datasets (LUNA16, SynthRAD2025)
docs/REPRODUCIBILITY.md  Step-by-step reproduction guide for the paper experiments
THIRD_PARTY.md         External method repositories and pinned commits
```

## Environment

- Python 3.11+, PyTorch (CUDA), NVIDIA GPU (the paper used a single consumer-grade GPU; INR querying is point-wise and memory-friendly)
- TIGRE toolbox for cone-beam projection/DRR simulation ([CERN/TIGRE](https://github.com/CERN/TIGRE), CUDA required)

```bash
pip install -r requirements.txt
```

The DeepSparse baseline additionally follows `configs/deepsparse/requirements-runtime.txt` and requires the external repositories listed in `THIRD_PARTY.md` (pinned commits included). External code is **not** redistributed.

## Data

### LUNA16 (public)

1. Download LUNA16 (subsets 0-9) from the [challenge site](https://luna16.grand-challenge.org/).
2. Build the DRR training/evaluation caches under the CBCT acquisition geometry with TIGRE (`tools/build_luna16_*` scripts, see `docs/REPRODUCIBILITY.md`).
3. Patient-level split registries are provided in `data_manifest/` so that the evaluation set is identical to the paper's.

### Real clinical head-and-neck CBCT

The real kV-projection/CBCT dataset was acquired clinically (retrospective study approved by the Ethics Committee of Fujian Medical University Cancer Hospital, No. K2024-050-01; informed consent waived). It **cannot be redistributed**; manifests referencing the clinical cohort are likewise excluded from this repository. The data are available from the corresponding author upon reasonable request.

## Reproducing the paper

See **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** for the full pipeline: preprocessing → training (all view configurations and augmentation settings) → evaluation against FDK / SART / DIF-Net / C2RV / DeepSparse / DuFal → setup-error restoration experiments → downstream lung segmentation → statistical analysis (paired bootstrap CIs, Holm-corrected Wilcoxon tests).

Unit tests (no data needed):

```bash
pytest
```

## Citation

If this code is useful to your research, please cite the paper above. The code itself is archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22845130.svg)](https://doi.org/10.5281/zenodo.22845130)


```bibtex
@article{dai2026casecbct,
  title   = {CaSE-CBCT: Capsule Self-Attention-Based Extremely Sparse-View CBCT Reconstruction},
  author  = {Dai, Yufan and Xie, Xin and Zhang, Jingjing and Yan, Hui and Li, Teng},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party repositories keep their original licenses (see `THIRD_PARTY.md`).
