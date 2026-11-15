# Solar Filament Segmentation Challenge 2026 — solution

Instance segmentation of solar filaments in GONG full-disk H-alpha images
([Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026),
IEEE BigData Cup 2026). Metric: Panoptic Quality (PQ) and Dice.

## Method in one paragraph

A U-Net with an ImageNet-pretrained EfficientNetV2-S encoder predicts a
per-pixel filament probability from the single-channel image (downscaled to
1024 px and normalized with on-disk statistics). It is trained on soft labels:
the per-pixel mean of the masks drawn by the image's annotators. The final
submission averages ten models (5 folds × 2 training seeds) over the eight
rotations and reflections of each test image. Instances are obtained by
thresholding the probability map at 0.5, taking 8-connected fragments,
grouping fragments that are connected through pixels with probability above 0.2
(hysteresis grouping, gaps up to 8 px bridged), and discarding candidates that
are unlikely to match an annotated filament: a gradient-boosted regressor
predicts each candidate's matched IoU from its probability, shape and stability
descriptors, and a candidate is kept when the prediction exceeds about PQ / 2,
the break-even point of the PQ formula. All post-processing parameters (and the
regressor) are fitted on out-of-fold predictions, scored exactly like the host's
evaluator (pooled PQ over annotator-image pairs, IoU > 0.5 matching, 2048 px).

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
| `weights/` | stored fold models (`<run>/best.pt`, `<run>/config.json`), learned filters (`instance_scorer*.joblib`) and manifests (`manifest.json` = final submission, `manifest_A.json` = single-seed variant: model list, test-time augmentation, post-processing, validation scores, fingerprint of the submitted file) |
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

`weights/manifest.json` is our final Kaggle submission: the 5-fold model trained with two seeds
(10 models). `weights/manifest_A.json` is the single-seed variant (5 models, also submitted);
run it by setting the environment variable `FIL_MANIFEST=manifest_A.json` before starting the
notebook. Both submitted files were written by this notebook from these weights, and each manifest
stores the SHA-256 of its file, which section 8 compares with. GPU kernels (and bf16 vs fp16 on
older GPUs) can round a few borderline pixels differently, so a re-run elsewhere may differ in a
handful of masks.

**From scratch** (caches, then training):

```bash
python src/prepare.py                     # 1024 px cache (images, masks, soft targets)
```

The final submission, step by step (bash; fold models take about 30 min each on an RTX 5060):

```bash
# 1. two 5-fold runs of the same model (seed 42 locally, seed 43 was run on Kaggle T4 GPUs with --bs 8)
for k in 0 1 2 3 4; do
  python src/train.py --name effv2ssoft_f$k --encoder tu-tf_efficientnetv2_s --fold $k \
      --soft --ema --epochs 65 --bs 4 --crop 768 --lr 3e-4 --val-every 5
  python src/train.py --name effv2ssoftk43_f$k --encoder tu-tf_efficientnetv2_s --fold $k \
      --soft --ema --epochs 65 --bs 8 --crop 768 --lr 3e-4 --val-every 5 --seed 43
done
# 2. out-of-fold maps with D4 test-time augmentation, then the two seeds averaged
for k in 0 1 2 3 4; do
  python src/infer.py --runs effv2ssoft_f$k --split val --fold $k --out effv2ssoftd4_f$k --tta d4
  python src/infer.py --runs effv2ssoftk43_f$k --split val --fold $k --out effv2ssoftk43d4_f$k --tta d4
done
python src/collect_oof.py --prefix effv2ssoftd4
python src/collect_oof.py --prefix effv2ssoftk43d4
python src/blend.py --preds effv2ssoftd4_oof effv2ssoftk43d4_oof --split val --out pairk43_oof
# 3. post-processing fitted on the out-of-fold maps: fixed rule (reference), then the learned filter
#    on hysteresis-grouped candidates (threshold 0.5, group_low 0.2, merge_px 8)
python src/tune.py --pred effv2ssoftd4_oof --fold -1
python src/instance_scorer.py --pred pairk43_oof --baseline effv2ssoftd4_oof \
    --threshold 0.5 --merge-px 8 --group-low 0.2   # -> preds/pairk43_oof/post_params_learned_g0.2.json
# 4. store the weights and parameters; the notebook then writes the submission from them
python src/export_weights.py --runs effv2ssoft_f0 effv2ssoft_f1 effv2ssoft_f2 effv2ssoft_f3 effv2ssoft_f4 \
    effv2ssoftk43_f0 effv2ssoftk43_f1 effv2ssoftk43_f2 effv2ssoftk43_f3 effv2ssoftk43_f4 \
    --post pairk43_oof --post-file post_params_learned_g0.2.json --submitted final_B_v1.csv --tta d4
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

Out-of-fold over the 707 training images (5-fold CV grouped by image; the learned
filter is cross-fitted, i.e. fold k is filtered by a model trained without fold k).
95% intervals come from 2,000 bootstrap resamples of images; ΔPQ is relative to the
row above (paired bootstrap).

| Configuration | PQ [95% CI] | Dice | ΔPQ [95% CI] | Public LB |
|---|---|---|---|---|
| ResNet34, one sample per annotator, flip TTA, fixed rule | 0.421 [0.412, 0.430] | 0.673 | | 0.36 (fold 0 only) |
| EfficientNetV2-S encoder | 0.430 [0.420, 0.439] | 0.682 | +0.008 [+0.004, +0.013] | |
| + D4 test-time augmentation | 0.433 [0.424, 0.442] | 0.680 | +0.004 [+0.001, +0.006] | |
| + soft labels, EMA, 65 epochs | 0.440 [0.431, 0.449] | 0.678 | +0.007 [+0.003, +0.011] | |
| + learned instance filter | 0.445 [0.436, 0.454] | 0.675 | +0.004 [+0.002, +0.007] | 0.39 |
| + hysteresis grouping (`manifest_A.json`, 5 models) | 0.448 [0.438, 0.457] | 0.672 | +0.003 [+0.001, +0.005] | 0.39 |
| + second seed (**final**, `manifest.json`, 10 models) | **0.449 [0.440, 0.458]** | 0.672 | +0.001 [−0.001, +0.004] | 0.39 |

For reference, two annotators of the same image agree at PQ 0.34 (Dice 0.61).
Inference for the 180 test images takes about 8 minutes on an RTX 5060
(10 models × 8 views, then post-processing).

## License

MIT (see `LICENSE`).
