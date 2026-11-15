# Solar Filament Segmentation Challenge 2026 — solution

Instance segmentation of solar filaments in GONG full-disk H-alpha images
([Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026),
IEEE BigData Cup 2026). Metric: Panoptic Quality (PQ) and Dice.

## Method in one paragraph

A U-Net with an ImageNet-pretrained encoder predicts a per-pixel filament
probability from the single-channel image (normalized with on-disk statistics).
Every annotation set is a training sample, so observations labeled by several
experts contribute each expert's view (label-noise regularization). Instances
are obtained by thresholding the probability map, taking 8-connected
components, optionally merging nearby fragments, and discarding components
that are unlikely to match an annotated filament. The discarding step is
either a fixed rule (small or not confident components) or a gradient-boosted
regressor that predicts each component's matched IoU from its probability,
shape and stability descriptors; a component is kept when the prediction
exceeds about PQ / 2, the break-even point of the PQ formula. All
post-processing parameters (and the regressor) are fitted on out-of-fold
predictions, scored exactly like the host's evaluator (pooled PQ over
annotator-image pairs, IoU > 0.5 matching, 2048 px).

## Repository layout

| File | Purpose |
|---|---|
| `src/common.py` | paths, COCO annotation loading, polygon rasterization (pycocotools), grouped CV folds |
| `src/prepare.py` | builds the resized image / mask cache, incl. annotator-mean (soft) targets |
| `src/dataset.py` | PyTorch dataset and augmentations |
| `src/model.py` | model construction, flip test-time augmentation |
| `src/train.py` | training with per-epoch validation (Dice, PQ); options for soft targets and weight EMA |
| `src/infer.py` | probability maps for validation or test images (multi-model averaging) |
| `src/postprocess.py` | probability map → filament instances (fixed or learned filter) |
| `src/instance_features.py` | descriptors of candidate instances (input of the learned filter) |
| `src/instance_scorer.py` | trains / tunes the learned filter on out-of-fold maps (cross-fitted by fold) |
| `src/metrics.py` | Panoptic Quality and Dice, matching the host's definition |
| `src/tune.py` | grid search of post-processing on out-of-fold predictions |
| `src/make_submission.py` | writes the submission CSV (compressed COCO RLE) |
| `src/annotator_agreement.py` | expert-vs-expert agreement (human reference) |
| `src/export_weights.py` | copies trained runs to `weights/` (fp16) and writes `weights/manifest.json` |
| `weights/` | stored fold models (`<run>/best.pt`, `<run>/config.json`), learned filter (`instance_scorer.joblib`) and `manifest.json` (model list, test-time augmentation, post-processing, validation scores, fingerprint of the submitted file) |
| `notebooks/pipeline.ipynb` | the whole pipeline step by step (data, preprocessing, model, inference, post-processing, validation), then the full reproduction of the submission from `weights/` |

## Reproduce

Requirements: Python 3.12, an NVIDIA GPU with ≥ 8 GB memory.

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Place the competition data at `data/MAGFiLO_1.0_Kaggle_2026/` (as unzipped from Kaggle).

**From the stored weights (no training):** open `notebooks/pipeline.ipynb` and run all cells.
Sections 1–7 illustrate every step on one image and also run on a CPU (about a minute);
section 8 regenerates the submission (`submissions/reproduced.csv`) from `weights/` and the
180 test images and runs only when a CUDA GPU is available. Headless:

```bash
jupyter nbconvert --to notebook --execute notebooks/pipeline.ipynb --output pipeline_executed.ipynb
```

`weights/manifest.json` is the final configuration (5 models). Our second selected Kaggle
submission adds the same model trained with a second seed (10 models, `weights/manifest_B.json`);
reproduce it by setting the environment variable `FIL_MANIFEST=manifest_B.json` before running the
notebook. Both submitted files were written by this notebook from these weights, and each manifest
stores the SHA-256 of its file, which section 8 compares with. GPU kernels (and bf16 vs fp16 on
older GPUs) can round a few borderline pixels differently, so a re-run elsewhere may differ in a
handful of masks.

**From scratch** (caches, then training):

```bash
python src/prepare.py                     # 1024 px cache
python src/prepare.py --size 2048         # native-resolution cache
# (training / inference commands of the final submission are listed below)
```

Training options (`python src/train.py --help`):

* `--soft`: one sample per observation, trained on the per-pixel mean of its
  annotators' masks (0, 1/3, 1/2, 2/3, 1) instead of one sample per
  annotator. Needs `python src/prepare.py --soft-only` (included in a full
  `prepare.py` run). The Dice loss is the Dice semimetric loss, minimized at
  prediction = soft target. Use ~65 instead of 40 epochs (fewer samples per epoch).
* `--ema [--ema-decay 0.998]`: validate and save an exponential moving average of the weights.
* `--select pq|combo`: keep the checkpoint with the best pooled PQ, or the best (Dice + PQ) / 2 (default).

Validation is always per annotation set, so all variants are scored the same way.

Post-processing is fitted on out-of-fold maps (`preds/<run>/val/`, all five folds):

```bash
python src/tune.py --pred r34_oof --fold -1            # fixed rule -> preds/r34_oof/post_params.json
python src/instance_scorer.py --pred r34_oof           # learned filter -> preds/r34_oof/post_params_learned.json
python src/make_submission.py --pred final --params preds/r34_oof/post_params_learned.json --out final
```

`instance_scorer.py` cross-fits the filter by fold and prints PQ / Dice per fold
next to the fixed rule; `--scorer-from <run>` applies a filter trained on another
run's maps (e.g. a single model's filter to an ensemble) and only re-tunes its threshold.

## Results

_To be filled in._

## License

MIT (see `LICENSE`).
