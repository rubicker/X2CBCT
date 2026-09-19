# Reproducing the paper experiments

All run configurations under `configs/` are self-contained (no YAML
inheritance): each file fixes the model, training, evaluation and runtime
settings of one experiment. Canonical invocations of the dataset-construction
steps are additionally recorded in the `command` field of every manifest under
`data_manifest/`.

## 0. Environment

```bash
conda create -n x2cbct python=3.11 -y
conda activate x2cbct
pip install -r requirements.txt        # + CUDA PyTorch build
# TIGRE: follow https://github.com/CERN/TIGRE (CUDA toolkit required)
pytest                                  # sanity check, no data required
```

## 1. LUNA16 (public data; DRR-based experiments, Fig./Tables of the main comparison)

1. Download LUNA16 subsets 0-9 and point the tools below at the extracted root.
2. Base volume preprocessing (HU windows, resampling):
   `tools/build_luna16_training_base_preprocess.py`
3. DRR generation and INR training-cache construction under the CBCT geometry
   (TIGRE): `tools/build_luna16_difnet_dataset.py`,
   `tools/derive_difnet_v1_training_cache.py`
4. Independent evaluation cache (149-case set used in the paper):
   `tools/build_luna16_independent_test_cache.py`
5. Split registries (identical to the paper):
   `data_manifest/luna16_patient_split_registry*.json`,
   `data_manifest/luna16_c2rv_split_registry.json`, and the LUNA16 manifests in
   `data_manifest/difnet_v1_si10_rot10_3x3_256/` (5/8/10-view, NoAug/Aug).

View configurations are fixed target gantry angles (5/8/10-view, nested sets),
identical across all compared methods.

## 2. Real clinical head-and-neck data

`tools/build_real_training_base_preprocess.py` and
`tools/build_real_independent_test_cache.py` build the training/test caches.
The clinical projections/volumes are **not** redistributable (see README);
the clinical manifests are therefore absent from this repository. Users with
data access can regenerate all clinical manifests with the same tools.

## 3. Training

| Model | Entry point |
|---|---|
| CaSE-CBCT (proposed) | `tools/train_difnet_v0b.py` |
| DeepSparse baseline | `tools/train_deepsparse.py` (external repo + `configs/deepsparse/*.yaml`) |

- 5/8/10-view and NoAug/Aug variants are selected by the manifest/config pair;
  see `configs/deepsparse/v*_luna16_{noaug,aug}.yaml` naming scheme.
- Ablation I (cross-view fusion: MLP / max / self-attention / DL-CCR) is
  controlled inside the capsule-routing module in
  `src/x2cbct_difnet/capsule_routing.py`.

## 4. Evaluation (main comparison)

```bash
python tools/test_difnet_v0b.py     # CaSE-CBCT
python tools/test_deepsparse.py     # DeepSparse
python tools/test_fdk_baseline.py   # FDK
python tools/test_sart_baseline.py  # SART
```

DIF-Net, C2RV, DIF-Gaussian and DuFal run from their official repositories
(see `THIRD_PARTY.md` for pinned commits); metrics are computed with the
shared PSNR/SSIM implementation in `src/x2cbct_difnet/metrics.py` so that all
methods are scored identically.

## 5. Setup-error restoration experiments (Ablation II / Table V)

1. Pose-synchronized augmentation manifests (3x3 S-I translation x yaw grid):
   `tools/build_difnet_v1_derivative_manifests.py`
2. Synthetic setup-error test cases (+10 mm / +10 deg):
   `tools/create_setup_error_synthetic_test_cases.py`
3. Registration-based measurement (reference CBCT vs. treatment-day CBCT vs.
   reconstruction): `tools/build_cbct_setup_error_pair_manifest.py` and the
   `setup_error_*` modules in `src/x2cbct_difnet/`
   (rigid registration on the spine ROI, S-I translation and yaw readout).
4. Clinical-characteristics statistics (245 registration pairs): report
   builders under `tools/rebuild_cbct_setup_error_*`.

## 6. Downstream lung segmentation (LUNA16)

`src/x2cbct_lungmask/` runs the fixed LungMask R231 segmenter on real vs.
reconstructed volumes (left/right lung Dice, ASD). Experiment configs:
`configs/lungmask/*.json`; the representative-case selection rule for the
qualitative figure is included.

## 7. Statistical analysis

Paired bootstrap confidence intervals (10,000 resamples; patient-clustered
bootstrap for the real dataset) and Holm-corrected two-sided Wilcoxon
signed-rank tests are implemented in the `tools/analyze_*` scripts; they
consume the per-case metric files produced by step 4/6.

## 8. Sanity checks

`tools/audit_*` scripts re-verify projection geometry, HU-window consistency
and self-projection intensity alignment — useful when adapting the pipeline to
a new acquisition geometry.
