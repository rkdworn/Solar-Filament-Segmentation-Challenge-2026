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
that are small or not confident. All post-processing parameters are chosen by
grid search on out-of-fold predictions, scored exactly like the host's
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
| `src/postprocess.py` | probability map → filament instances |
| `src/metrics.py` | Panoptic Quality and Dice, matching the host's definition |
| `src/tune.py` | grid search of post-processing on out-of-fold predictions |
| `src/make_submission.py` | writes the submission CSV (compressed COCO RLE) |
| `src/annotator_agreement.py` | expert-vs-expert agreement (human reference) |

## Reproduce

Requirements: Python 3.12, an NVIDIA GPU with ≥ 8 GB memory.

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Place the competition data at `data/MAGFiLO_1.0_Kaggle_2026/` (as unzipped from Kaggle), then:

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

## Results

_To be filled in._

## License

MIT (see `LICENSE`).
