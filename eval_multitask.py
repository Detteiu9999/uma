"""Exploratory ranking + independent binary classifier blends, not multitask learning.

Three separately trained classifiers share neither parameters nor a joint objective.
Early stopping, temperature fitting, weight search and reporting reuse validation
labels, so results are optimistic tuning diagnostics, not adoption evidence.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import predict_model as pm
from evaluate_probs import logloss, valid_finish


def race_z(scores, groups):
    df = pd.DataFrame({"g": groups, "s": scores})
    return df.groupby("g")["s"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    ).to_numpy()


def main():
    print("EXPLORATORY ONLY: independent classifier blend, not proven multitask learning.")
    print("Same validation labels tune early stopping, temperatures and blend weights, "
          "then report metrics: optimistic, not adoption evidence. No walk-forward or OOF claim.")
    print("Production feature preparation is reused, not refitted inside nested folds; "
          "this script does not establish a leakage-free pipeline.")
    if pm.TIME_SPLIT_YEAR is None:
        raise SystemExit("A fixed TIME_SPLIT_YEAR is required to match production target encoding.")
    pm.DEVICE = pm.detect_device()
    print("device:", pm.DEVICE)
    train_df = pm.load_training_frame()
    codes = train_df["URLコード"].astype(str)
    if not codes.str.fullmatch(r"[0-9]{10}").all():
        raise SystemExit("Invalid year-inclusive race codes")
    valid = []
    for code, race in train_df.groupby("URLコード", sort=False):
        nums = pd.to_numeric(race["馬番"], errors="coerce").to_numpy(dtype=float)
        if valid_finish(nums, race["_finish"].to_numpy(dtype=float)):
            valid.append(code)
    excluded = train_df["URLコード"].nunique() - len(valid)
    train_df = train_df[train_df["URLコード"].isin(valid)].reset_index(drop=True)
    print(f"Invalid/ambiguous labeled races excluded: {excluded}; "
          "upstream loading may already have removed unranked horses.")
    years = pd.to_numeric(train_df["URLコード"].astype(str).str[:2])
    tr = np.flatnonzero(years.to_numpy() < pm.TIME_SPLIT_YEAR)
    va = np.flatnonzero(years.to_numpy() >= pm.TIME_SPLIT_YEAR)
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("Fixed year split is empty; refusing a target-encoding-incompatible fallback")
    finish = train_df["_finish"].reset_index(drop=True)
    X_enc, _, _, _, groups = pm.prepare_training_matrix(train_df)
    X_enc = X_enc.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    label = pm.make_rank_label(finish, groups)
    qid = pd.factorize(groups)[0]
    print(f"Configured split: code years < {pm.TIME_SPLIT_YEAR} train, "
          f">= {pm.TIME_SPLIT_YEAR} validation/tuning; "
          f"{groups.iloc[tr].nunique()} / {groups.iloc[va].nunique()} races")

    dtr = xgb.DMatrix(X_enc.iloc[tr], label=label.iloc[tr], qid=qid[tr], enable_categorical=True)
    dva = xgb.DMatrix(X_enc.iloc[va], label=label.iloc[va], qid=qid[va], enable_categorical=True)
    rank_model = xgb.train(pm.xgb_params(), dtr, num_boost_round=pm.MAX_ROUNDS,
                           evals=[(dva, "valid")], early_stopping_rounds=pm.EARLY_STOP,
                           verbose_eval=False)
    s_rank = rank_model.predict(dva, iteration_range=(0, rank_model.best_iteration + 1))
    if not np.isfinite(s_rank).all():
        raise SystemExit("Invalid ranking predictions; no metrics reported")
    f_va = finish.iloc[va].to_numpy()
    g_va = groups.iloc[va].to_numpy()
    temp, _ = pm.fit_temperature(s_rank, f_va, g_va)
    p1_base = pm.softmax_by_group(s_rank, g_va, temp)
    h1_base, h3_base, _ = pm.evaluate_ranking(s_rank, f_va, g_va)
    print(f"[tuning baseline] top-pick win={h1_base:.4f} top3={h3_base:.4f} "
          f"win LL={logloss(p1_base, f_va == 1):.4f}")

    classifier_params = dict(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=pm.DEVICE,
        learning_rate=0.05, max_depth=5, min_child_weight=30, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
        random_state=pm.RANDOM_SEED, verbosity=0,
    )
    z_classifiers = {}
    for k, name in ((1, "win"), (2, "top2"), (3, "top3")):
        y_tr = (finish.iloc[tr] <= k).astype(int)
        y_va = (finish.iloc[va] <= k).astype(int)
        mtr = xgb.DMatrix(X_enc.iloc[tr], label=y_tr, enable_categorical=True)
        mva = xgb.DMatrix(X_enc.iloc[va], label=y_va, enable_categorical=True)
        model = xgb.train(classifier_params, mtr, num_boost_round=3000,
                          evals=[(mva, "valid")], early_stopping_rounds=100, verbose_eval=False)
        p = model.predict(mva, iteration_range=(0, model.best_iteration + 1))
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise SystemExit("Invalid classifier probabilities; no blend metrics reported")
        print(f"[independent {name}; tuning] best_iter={model.best_iteration + 1} "
              f"LL={logloss(p, y_va):.4f}")
        clipped = np.clip(p, 1e-9, 1 - 1e-9)
        z_classifiers[k] = race_z(np.log(clipped / (1 - clipped)), g_va)

    z_rank = race_z(s_rank, g_va)
    best_hit, best_weights, selected = h1_base, None, s_rank
    for w in np.linspace(0.05, 0.5, 10):
        blend = (1 - w) * z_rank + w * z_classifiers[1]
        hit, _, _ = pm.evaluate_ranking(blend, f_va, g_va)
        if hit > best_hit:
            best_hit, best_weights, selected = hit, (float(w), 0.0), blend
    for w1 in np.linspace(0.05, 0.4, 8):
        for w23 in (0.05, 0.1, 0.15):
            blend = ((1 - w1 - 2 * w23) * z_rank + w1 * z_classifiers[1]
                     + w23 * z_classifiers[2] + w23 * z_classifiers[3])
            hit, _, _ = pm.evaluate_ranking(blend, f_va, g_va)
            if hit > best_hit:
                best_hit, best_weights, selected = hit, (float(w1), w23), blend
    print(f"[same-validation search] selected w1,w23={best_weights} "
          f"win hit={best_hit:.4f} (baseline {h1_base:.4f})")
    if best_weights is None:
        print("No improvement in this tuning search; no production adoption/rejection conclusion.")
        return
    blend_temp, _ = pm.fit_temperature(selected, f_va, g_va)
    baseline_losses, blend_losses = [], []
    for group in np.unique(g_va):
        mask = g_va == group
        base_probs = pm.pl_topk_probs(s_rank[mask], temp)
        blend_probs = pm.pl_topk_probs(selected[mask], blend_temp)
        baseline_losses.append([logloss(p, f_va[mask] <= k)
                                for k, p in enumerate(base_probs, 1)])
        blend_losses.append([logloss(p, f_va[mask] <= k)
                             for k, p in enumerate(blend_probs, 1)])
    print("[same-validation; race-mean top1/top2/top3 LL] baseline:",
          np.mean(baseline_losses, axis=0).round(4), "independent classifier blend:",
          np.mean(blend_losses, axis=0).round(4))
    print("Optimistic exploratory diagnostics only; an untouched evaluation with fold-local "
          "preprocessing and independently selected rounds/weights is still required.")


if __name__ == "__main__":
    main()
