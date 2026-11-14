"""Train and tune the learned instance filter on out-of-fold (OOF) maps.

The learned filter (``postprocess.py``) drops candidate instances whose
predicted *matched IoU* is not above ``min_score``. This script fits that
predictor honestly and reports its effect:

1. candidates  For every training observation, the components of its OOF map
               (same threshold / merging as at test time) with their
               descriptors (``instance_features.py``) and, per annotation set,
               their matched IoU (IoU with a ground-truth filament if > 0.5,
               else 0), number of matched filaments and overlap with the
               set's filament union. Target = matched IoU averaged over the
               observation's annotation sets.
2. cross-fit   For each fold k, a gradient-boosted regressor trained on the
               other folds scores fold k, so every reported number uses
               scores the model has not been trained on.
3. tune        ``min_score`` maximising pooled PQ of the OOF selection, also
               chosen on four folds and applied to the fifth ("nested"),
               reported per fold next to the current rule.
4. save        The fold models and a final model fitted on all folds
               (``preds/<pred>/instance_scorer/fold<k>.joblib``, ``all.joblib``)
               and the parameters ``preds/<pred>/post_params_learned.json``.

PQ and Dice of any selection are computed exactly from the candidate table:
candidates are pixel-disjoint and a ground-truth filament can match at most
one of them (IoU > 0.5), so per-candidate statistics add up. ``--verify``
re-scores the OOF maps end to end with ``postprocess.to_instances`` and the
fold models as a check. The table is cached in ``preds/<pred>/``.

Examples:
    .venv\\Scripts\\python.exe src\\instance_scorer.py --pred r34_oof --verify
    .venv\\Scripts\\python.exe src\\instance_scorer.py --pred blend_oof --scorer-from r34_oof --threshold 0.55 --merge-px 16
    .venv\\Scripts\\python.exe src\\make_submission.py --pred final --params preds\\r34_oof\\post_params_learned.json --out v2
"""
from __future__ import annotations

import argparse
import json

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

from common import FULL_SIZE, PRED_DIR, ROOT, instance_masks, load_annotations
from instance_features import FEATURES
from metrics import MATCH_IOU, pq_stats, to_flat
from postprocess import (PostParams, candidate_features, label_components, score_candidates, to_instances,
                         upsample_prob)

STEP = 0.05          # threshold offset of the stability features
MIN_AREA = 50        # smaller components are never scored (the smallest annotated filament has ~95 px)
MIN_SCORES = np.round(np.arange(0.10, 0.36, 0.02), 2)
# Shallow trees, averaged over bootstrap replicas of the observations: across the r34 and r34d4 OOF
# maps (each scored by a model trained on either) this was the most stable setting.
GBM = dict(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=40, l2_regularization=1.0,
           early_stopping=False)
N_BAGS = 5


# --------------------------------------------------------------------------- #
# 1. Candidate table
# --------------------------------------------------------------------------- #
def load_prob(pred: str, stem: str) -> np.ndarray:
    p8 = cv2.imread(str(PRED_DIR / pred / "val" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if p8 is None:
        raise FileNotFoundError(f"missing OOF map {pred}/val/{stem}.png")
    return upsample_prob(p8.astype(np.float32) / 255.0, FULL_SIZE)


def annotation_set(polygons: dict, image_id: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Ground-truth filaments of one annotation set (flat pixel indices at 2048 px) and their union."""
    gts = [to_flat(m) for m in instance_masks(polygons[image_id], FULL_SIZE)]
    union = np.unique(np.concatenate(gts)) if gts else np.zeros(0, np.int64)
    return gts, union


def label_map_stats(labels: np.ndarray, gts: list[np.ndarray], union: np.ndarray) -> dict:
    """Host PQ statistics and Dice of a label map against one annotation set."""
    s = pq_stats(labels, gts)
    fg = labels.ravel() > 0
    den = fg.sum() + union.size
    return dict(iou=s.iou_sum, tp=s.tp, fp=s.fp, fn=s.fn, dice=2.0 * fg[union].sum() / den if den else 1.0)


def set_targets(labels: np.ndarray, gts: list[np.ndarray], union: np.ndarray, ids: np.ndarray):
    """Per candidate id: matched IoU, number of matched filaments and overlap with the filament union."""
    flat = labels.ravel()
    n = int(flat.max()) + 1
    area = np.bincount(flat, minlength=n).astype(np.float64)
    miou, ntp = np.zeros(n), np.zeros(n)
    for g in gts:
        inter = np.bincount(flat[g], minlength=n).astype(np.float64)
        iou = inter / (g.size + area - inter)
        iou[0] = 0
        hit = iou > MATCH_IOU  # at most one candidate per filament
        miou[hit] += iou[hit]
        ntp[hit] += 1
    overlap = np.bincount(flat[union], minlength=n)
    return miou[ids], ntp[ids], overlap[ids]


def build_table(pred: str, index: pd.DataFrame, polygons: dict, threshold: float, merge_px: int,
                baseline: PostParams) -> dict:
    """Candidates (features), candidate x annotation-set targets, and annotation sets (with the baseline's stats)."""
    cands, pairs, sets = [], [], []
    n_cand = 0
    for stem, g in tqdm(index.groupby("stem"), desc=f"candidates {pred}"):
        prob = load_prob(pred, stem)
        labels = label_components(prob, threshold, merge_px)
        feats = candidate_features(labels, prob, threshold, merge_px, STEP, MIN_AREA)
        ids = feats.index.to_numpy()
        base = to_instances(prob, baseline)
        miou_sum = np.zeros(len(ids))
        for image_id, fold in zip(g["image_id"], g["fold"]):
            gts, union = annotation_set(polygons, image_id)
            miou, ntp, overlap = set_targets(labels, gts, union, ids)
            miou_sum += miou
            set_id = len(sets)
            pairs.append(pd.DataFrame(dict(cand=np.arange(n_cand, n_cand + len(ids)), set_id=set_id, miou=miou,
                                           ntp=ntp, overlap=overlap)))
            base_stats = {f"base_{k}": v for k, v in label_map_stats(base, gts, union).items()}
            sets.append(dict(set_id=set_id, image_id=image_id, stem=stem, fold=int(fold), n_gt=len(gts),
                             union_px=union.size, **base_stats))
        feats = feats.reset_index()
        feats["stem"], feats["fold"], feats["n_sets"] = stem, int(g["fold"].iloc[0]), len(g)
        feats["target"] = miou_sum / len(g)
        cands.append(feats)
        n_cand += len(ids)
    return dict(cands=pd.concat(cands, ignore_index=True), pairs=pd.concat(pairs, ignore_index=True),
                sets=pd.DataFrame(sets))


# --------------------------------------------------------------------------- #
# Exact PQ / Dice of a selection
# --------------------------------------------------------------------------- #
def selection_stats(table: dict, keep: np.ndarray) -> pd.DataFrame:
    """Per annotation set: IoU sum, TP, FP, FN and Dice when the candidates in ``keep`` are reported."""
    cands, pairs, sets = table["cands"], table["pairs"], table["sets"]
    kept = pairs[keep[pairs["cand"].to_numpy()]]
    per_set = kept.assign(matched=kept["ntp"] > 0).groupby("set_id")[["miou", "ntp", "matched", "overlap"]].sum()
    per_obs = cands[keep].groupby("stem").agg(n_kept=("area", "size"), kept_px=("area", "sum"))
    out = sets[["set_id", "stem", "fold", "n_gt", "union_px"]].join(per_set, on="set_id").join(per_obs, on="stem")
    out = out.fillna(0)
    den = out["kept_px"] + out["union_px"]
    return pd.DataFrame(dict(stem=out["stem"], fold=out["fold"], iou=out["miou"], tp=out["ntp"],
                             fp=out["n_kept"] - out["matched"], fn=out["n_gt"] - out["ntp"],
                             dice=np.where(den > 0, 2 * out["overlap"] / np.maximum(den, 1), 1.0)))


def baseline_stats(table: dict) -> pd.DataFrame:
    s = table["sets"]
    return s[["stem", "fold", "base_iou", "base_tp", "base_fp", "base_fn", "base_dice"]].rename(
        columns=lambda c: c.removeprefix("base_"))


def pooled_pq(st: pd.DataFrame) -> float:
    return st["iou"].sum() / (st["tp"].sum() + 0.5 * st["fp"].sum() + 0.5 * st["fn"].sum())


def summary(st: pd.DataFrame) -> dict:
    folds = [pooled_pq(f) for _, f in st.groupby("fold")]
    return dict(pq_pooled=pooled_pq(st), dice=float(st["dice"].mean()), pq_folds=[round(v, 4) for v in folds],
                pq_fold_mean=float(np.mean(folds)), pq_fold_std=float(np.std(folds, ddof=1)),
                tp=int(st["tp"].sum()), fp=int(st["fp"].sum()), fn=int(st["fn"].sum()))


def paired_bootstrap(new: pd.DataFrame, old: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """PQ difference new - old with a 95% interval, resampling observations (as compare.py does)."""
    cols = ["iou", "tp", "fp", "fn"]
    per_obs = new.groupby("stem")[cols].sum()
    a, b = per_obs.to_numpy(), old.groupby("stem")[cols].sum().loc[per_obs.index].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))

    def pq(x):
        s = x.sum(axis=1)
        return s[:, 0] / (s[:, 1] + 0.5 * s[:, 2] + 0.5 * s[:, 3])
    d = pq(a[idx]) - pq(b[idx])
    return dict(dpq=float(d.mean()), ci95=[round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)],
                p_better=float((d > 0).mean()))


# --------------------------------------------------------------------------- #
# 2. Cross-fitting and 3. tuning
# --------------------------------------------------------------------------- #
def fit_scorer(cands: pd.DataFrame, threshold: float, merge_px: int, trained_on: str, seed: int = 0) -> dict:
    """Fit the bagged regressor and bundle it with the candidate definition it is valid for.

    Squared loss -> the expected matched IoU. Each annotation set counts once
    in pooled PQ, so an observation's candidates are weighted by its number
    of annotation sets. Each bag resamples whole observations (candidates of
    one image are correlated), implemented as integer sample weights.
    """
    x, y = cands[FEATURES].to_numpy(np.float32), cands["target"].to_numpy()
    stems = cands["stem"].unique()
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(N_BAGS):
        copies = pd.Series(rng.choice(stems, size=len(stems))).value_counts()
        weight = cands["n_sets"].to_numpy() * cands["stem"].map(copies).fillna(0).to_numpy()
        models.append(HistGradientBoostingRegressor(**GBM).fit(x, y, sample_weight=weight))
    return dict(models=models, features=FEATURES, threshold=threshold, merge_px=merge_px, min_area=MIN_AREA,
                step=STEP, trained_on=trained_on, sklearn=sklearn.__version__)


def crossfit_scores(cands: pd.DataFrame, folds: list[int], directory, train: bool, threshold: float,
                    merge_px: int, pred: str) -> np.ndarray:
    """Score fold k with the model trained without fold k (trained here, or loaded from ``directory``)."""
    score = np.zeros(len(cands))
    for k in folds:
        test = (cands["fold"] == k).to_numpy()
        path = directory / f"fold{k}.joblib"
        if train:
            joblib.dump(fit_scorer(cands[~test], threshold, merge_px, f"{pred} without fold {k}"), path)
        bundle = joblib.load(path)
        if (bundle["threshold"], bundle["merge_px"]) != (threshold, merge_px):
            raise ValueError(f"{path} was trained for other candidates ({bundle['threshold']}, {bundle['merge_px']})")
        score[test] = score_candidates(bundle, cands[test])
    return score


def tune_min_score(table: dict, score: np.ndarray, objective: str):
    """Best min_score on all folds, plus the nested estimate (chosen on four folds, applied to the fifth).

    Returns the chosen min_score, per-set statistics of both selections and the PQ / Dice curve.
    """
    stats = {t: selection_stats(table, score > t) for t in MIN_SCORES}

    def value(st):
        return pooled_pq(st) if objective == "pq" else (pooled_pq(st) + st["dice"].mean()) / 2

    best = max(MIN_SCORES, key=lambda t: value(stats[t]))
    nested = []
    for k in sorted(table["sets"]["fold"].unique()):
        tk = max(MIN_SCORES, key=lambda t: value(stats[t][stats[t]["fold"] != k]))
        nested.append(stats[tk][stats[tk]["fold"] == k])
    curve = pd.DataFrame([dict(min_score=t, **summary(stats[t])) for t in MIN_SCORES])
    return float(best), dict(chosen=stats[best], nested=pd.concat(nested)), curve


def scorer_dir(pred: str):
    return PRED_DIR / pred / "instance_scorer"


def verify(pred: str, source: str, index: pd.DataFrame, polygons: dict, params: PostParams,
           folds: list[int]) -> dict:
    """End-to-end check: ``to_instances`` with each fold's model (trained on ``source``) on that fold's maps."""
    stats = []
    for k in folds:
        p = PostParams(**{**params.as_dict(), "scorer": str(scorer_dir(source) / f"fold{k}.joblib")})
        for stem, g in tqdm(index[index["fold"] == k].groupby("stem"), desc=f"verify fold {k}"):
            labels = to_instances(load_prob(pred, stem), p)
            for image_id in g["image_id"]:
                stats.append(dict(stem=stem, fold=k, **label_map_stats(labels, *annotation_set(polygons, image_id))))
    return summary(pd.DataFrame(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="name under preds/ holding OOF maps in val/")
    ap.add_argument("--baseline", default=None,
                    help="run under preds/ whose post_params.json is the rule to compare with (default: --pred)")
    ap.add_argument("--threshold", type=float, default=None, help="candidate threshold (default: baseline's)")
    ap.add_argument("--merge-px", type=int, default=None, help="candidate merge radius (default: baseline's)")
    ap.add_argument("--scorer-from", default=None,
                    help="reuse the fold models / final model trained on another run instead of training")
    ap.add_argument("--objective", default="pq", choices=["pq", "combo"], help="combo = (PQ + Dice) / 2")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="restrict to these folds (quick look)")
    ap.add_argument("--rebuild", action="store_true", help="recompute the cached candidate table")
    ap.add_argument("--verify", action="store_true", help="also re-score the OOF maps end to end")
    args = ap.parse_args()

    baseline_run = args.baseline or args.pred
    baseline = PostParams(**json.loads((PRED_DIR / baseline_run / "post_params.json").read_text())["params"])
    threshold = baseline.threshold if args.threshold is None else args.threshold
    merge_px = baseline.merge_px if args.merge_px is None else args.merge_px

    index, polygons = load_annotations()
    if args.folds is not None:
        index = index[index["fold"].isin(args.folds)]
    folds = sorted(index["fold"].unique())

    # 1. candidate table (cached: building it takes a few minutes)
    settings = dict(threshold=threshold, merge_px=merge_px, min_area=MIN_AREA, step=STEP,
                    baseline=baseline.as_dict(), folds=folds, features=FEATURES)
    cache = PRED_DIR / args.pred / f"instance_candidates_t{threshold:g}_m{merge_px}.joblib"
    table = joblib.load(cache) if cache.exists() and not args.rebuild else None
    if table is None or table["settings"] != settings:
        table = build_table(args.pred, index, polygons, threshold, merge_px, baseline)
        table["settings"] = settings
        joblib.dump(table, cache)
    cands = table["cands"]
    print(f"{len(cands)} candidates in {cands['stem'].nunique()} observations, "
          f"{len(table['sets'])} annotation sets")

    # 2. cross-fitted scores (fold k scored by a model that never saw fold k)
    source = args.scorer_from or args.pred
    train = args.scorer_from is None
    scorer_dir(source).mkdir(parents=True, exist_ok=True)
    score = crossfit_scores(cands, folds, scorer_dir(source), train, threshold, merge_px, args.pred)

    # 3. tune min_score and compare with the current rule
    min_score, res, curve = tune_min_score(table, score, args.objective)
    base_sets = baseline_stats(table)
    base, chosen, nested = summary(base_sets), summary(res["chosen"]), summary(res["nested"])
    pd.set_option("display.width", 200)
    print(curve[["min_score", "pq_pooled", "dice", "pq_fold_mean", "pq_fold_std", "tp", "fp", "fn"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    rows = {f"current rule ({baseline_run})": base, f"learned, min_score={min_score:.2f}": chosen,
            "learned, min_score tuned on other folds": nested}
    print(pd.DataFrame(rows).T[["pq_pooled", "dice", "pq_fold_mean", "pq_fold_std", "pq_folds", "tp", "fp", "fn"]]
          .to_string())
    better = sum(a > b for a, b in zip(nested["pq_folds"], base["pq_folds"]))
    boot = paired_bootstrap(res["nested"], base_sets)
    print(f"nested learned rule beats the current rule in {better}/{len(folds)} folds; "
          f"paired bootstrap dPQ {boot['dpq']:+.4f} {boot['ci95']}, P(better) = {boot['p_better']:.3f}")

    # 4. final model (all folds) and parameters for inference
    final = scorer_dir(source) / "all.joblib"
    if train:
        joblib.dump(fit_scorer(cands, threshold, merge_px, f"{args.pred} (all folds)"), final)
    params = PostParams(threshold=threshold, merge_px=merge_px, min_area=MIN_AREA,
                        scorer=final.relative_to(ROOT).as_posix(), min_score=min_score)
    out = dict(params=params.as_dict(), objective=args.objective, **chosen, nested=nested,
               baseline=dict(run=baseline_run, **base), bootstrap_nested_vs_baseline=boot)
    if args.verify:
        out["verify"] = verify(args.pred, source, index, polygons, params, folds)
        print("end-to-end check:", out["verify"])
    if args.folds is None:  # only a full run defines the parameters to use
        (PRED_DIR / args.pred / "post_params_learned.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["params"], indent=2))


if __name__ == "__main__":
    main()
