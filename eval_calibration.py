"""Exploratory p1 calibration on a deterministic, non-temporal race-group holdout.

Only p1 is fitted. Normalized calibrated strengths define PL top2/top3 marginals.
This is not OOF, a temporal validation, or evidence for production adoption.
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from evaluate_probs import RACE_COLUMNS, brier, logloss, valid_race
from predict_model import pl_topk_probs


def validated_frame(df):
    required = RACE_COLUMNS + ["馬番", "着順", "p1", "p2", "p3"]
    if not set(required) <= set(df.columns):
        raise ValueError("Year-inclusive export required: run evaluate_probs.py --save-heldout")
    df = df[required].apply(pd.to_numeric, errors="coerce")
    keys = df[RACE_COLUMNS].to_numpy(dtype=float)
    if (not np.isfinite(keys).all() or not (keys == np.floor(keys)).all()
            or not df["年"].between(2000, 2099).all()
            or not df["競馬場コード"].between(1, 10).all()
            or (df[["回", "日", "レース"]] < 1).any().any()):
        raise ValueError("Missing or invalid year-inclusive race keys; regenerate the export")
    df[RACE_COLUMNS] = df[RACE_COLUMNS].astype(int)
    valid = []
    skipped = 0
    for _, race in df.groupby(RACE_COLUMNS, sort=True):
        if valid_race(*(race[col].to_numpy() for col in ("馬番", "着順", "p1", "p2", "p3"))):
            race = race.copy()
            race["p1"] /= race["p1"].sum()
            valid.append(race)
        else:
            skipped += 1
    print(f"Invalid/ambiguous races excluded: {skipped}")
    if not valid:
        raise ValueError("No valid races")
    return pd.concat(valid).sort_values(RACE_COLUMNS + ["馬番"]).reset_index(drop=True)


def split_races(df, seed=42):
    keys = list(df[RACE_COLUMNS].itertuples(index=False, name=None))
    unique = sorted(set(keys), key=lambda key: (
        hashlib.sha256(f"{seed}:{','.join(map(str, key))}".encode("ascii")).digest(), key
    ))
    if len(unique) < 2:
        raise ValueError("At least two valid races are needed for a race-group holdout")
    split = max(1, min(len(unique) - 1, len(unique) * 2 // 3))
    fit_keys = set(unique[:split])
    mask = np.array([key in fit_keys for key in keys])
    return df.loc[mask].copy(), df.loc[~mask].copy()


def calibrated_probabilities(eva, iso):
    output = np.empty((len(eva), 3), dtype=float)
    for positions in eva.groupby(RACE_COLUMNS, sort=False).indices.values():
        strengths = iso.predict(eva.iloc[positions]["p1"].to_numpy())
        if not np.isfinite(strengths).all() or (strengths < 0).any() or (strengths > 1).any():
            raise ValueError("Invalid calibrated strengths")
        strengths = np.maximum(strengths, 1e-12)
        strengths /= strengths.sum()
        output[positions] = np.column_stack(pl_topk_probs(np.log(strengths), 1.0))
    return output


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=os.path.join(base, "predictions", "_heldout_probs.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print("EXPLORATORY ONLY: non-temporal race-group holdout, not adoption evidence or OOF.")
    print("Base-model training provenance is unverified; horses/meetings may overlap across groups. "
          "Only the calibration fit is held out from the evaluation races.")
    try:
        df = validated_frame(pd.read_csv(args.input, encoding="utf-8-sig"))
        fit, eva = split_races(df, args.seed)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        ap.error(str(exc))
    print(f"Deterministic hash split (seed={args.seed}): "
          f"fit={fit.groupby(RACE_COLUMNS).ngroups} races/{len(fit)} rows; "
          f"heldout={eva.groupby(RACE_COLUMNS).ngroups} races/{len(eva)} rows")
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(fit["p1"].to_numpy(), (fit["着順"] == 1).to_numpy(dtype=float))
    calibrated = calibrated_probabilities(eva, iso)
    print("p1-only isotonic fit; race-normalized strengths -> production PL top1/top2/top3. "
          "No independent top2/top3 calibration. Metrics are horse-weighted.")
    for k in (1, 2, 3):
        y = (eva["着順"] <= k).to_numpy(dtype=float)
        before = eva[f"p{k}"].to_numpy()
        after = calibrated[:, k - 1]
        print(f"Top{k}: stored LogLoss={logloss(before, y):.4f} Brier={brier(before, y):.4f} | "
              f"calibrated PL LogLoss={logloss(after, y):.4f} Brier={brier(after, y):.4f}")
    print("Top2/top3 differences include replacing stored marginals with PL; "
          "an untouched prospective evaluation is still required.")


if __name__ == "__main__":
    main()
