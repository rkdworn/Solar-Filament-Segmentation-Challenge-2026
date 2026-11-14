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
| `src/prepare.py` | builds the resized image / mask cache |
| `src/dataset.py` | PyTorch dataset and augmentations |
| `src/model.py` | model construction, flip test-time augmentation |
| `src/train.py` | training with per-epoch validation (Dice, PQ) |
| `src/infer.py` | probability maps for validation or test images (multi-model averaging) |
| `src/postprocess.py` | probability map → filament instances (fixed or learned filter) |
| `src/instance_features.py` | descriptors of candidate instances (input of the learned filter) |
| `src/instance_scorer.py` | trains / tunes the learned filter on out-of-fold maps (cross-fitted by fold) |
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
