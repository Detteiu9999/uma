"""Exploratory checks of stored predictions, not OOF or adoption evidence.

Use --save-heldout to export matched predictions for exploratory calibration.
The export name does not certify the base model's training-data provenance.
"""
import argparse
import glob
import os
import re
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from suggest_bets import (
    harville_exacta, harville_quinella, harville_trifecta,
    harville_trio, harville_wide,
)

PLACE_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}
RACE_COLUMNS = ["年", "競馬場コード", "回", "日", "レース"]


def indep_quinella(p2, a, b):
    return p2[a] * p2[b]


def indep_wide(p3, a, b):
    return p3[a] * p3[b]


def indep_exacta(p1, p2, a, b):
    return p1[a] * p2[b]


def indep_trio(p3, a, b, c):
    return p3[a] * p3[b] * p3[c]


def indep_trifecta(p1, p2, p3, a, b, c):
    return p1[a] * p2[b] * p3[c]


def logloss(probs, hits):
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(hits, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(probs, hits):
    return float(np.mean((np.asarray(probs, dtype=float) - np.asarray(hits, dtype=float)) ** 2))


def calibration_table(probs, hits, n_bins=10):
    df = pd.DataFrame({"p": probs, "y": hits})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return g.dropna()


def valid_finish(nums, finishes):
    nums = np.asarray(nums, dtype=float)
    finishes = np.asarray(finishes, dtype=float)
    return (
        nums.ndim == 1 and finishes.shape == nums.shape and len(nums) >= 3
        and np.isfinite(nums).all() and np.isfinite(finishes).all()
        and (nums > 0).all() and (nums == np.floor(nums)).all()
        and len(np.unique(nums)) == len(nums)
        and (finishes >= 1).all() and (finishes < 99).all()
        and (finishes == np.floor(finishes)).all()
        and all(np.count_nonzero(finishes == k) == 1 for k in (1, 2, 3))
    )


def valid_race(nums, finishes, p1, p2, p3):
    if not valid_finish(nums, finishes):
        return False
    probs = [np.asarray(p, dtype=float) for p in (p1, p2, p3)]
    if any(p.shape != np.asarray(nums).shape or not np.isfinite(p).all()
           or (p < 0).any() or (p > 1).any() for p in probs):
        return False
    p1, p2, p3 = probs
    return (np.isclose(p1.sum(), 1, atol=1e-4, rtol=0)
            and np.count_nonzero(p1) >= 3
            and (p1 <= p2 + 1e-6).all() and (p2 <= p3 + 1e-6).all())


def result_key(df):
    columns = ["年", "競馬場", "回", "日", "レース"]
    values = df[columns].apply(pd.to_numeric, errors="raise")
    if values.empty or values.isna().any().any() or not (values.nunique() == 1).all():
        raise ValueError("Missing or mixed race identity")
    first = values.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(first).all() or not (first == np.floor(first)).all():
        raise ValueError("Invalid race identity")
    key = tuple(int(x) for x in first)
    if not 2000 <= key[0] <= 2099 or key[1] not in PLACE_NAMES or min(key[2:]) < 1:
        raise ValueError("Invalid race identity")
    if "URLコード" in df:
        code = f"{key[0] % 100:02d}{key[1]:02d}{key[2]:02d}{key[3]:02d}{key[4]:02d}"
        if not df["URLコード"].astype(str).str.strip().eq(code).all():
            raise ValueError("Race code disagrees with metadata")
    return key


def load_kekka(kekka_dir):
    results, seen, rejected = {}, set(), set()
    skipped = 0
    for path in sorted(glob.glob(os.path.join(kekka_dir, "horse_racing_data_*.csv"))):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            key = result_key(df)
            if key in seen:
                rejected.add(key)
                raise ValueError("Duplicate result race")
            seen.add(key)
            nums = pd.to_numeric(df["馬番"], errors="raise").to_numpy(dtype=float)
            finishes = pd.to_numeric(
                df["着順"].str.extract(r"^\s*(\d+)着", expand=False), errors="coerce"
            ).to_numpy(dtype=float)
            if not valid_finish(nums, finishes):
                raise ValueError("Incomplete/ambiguous finishes or duplicate horses")
            results[key] = dict(zip(nums.astype(int), finishes.astype(int)))
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            skipped += 1
    for key in rejected:
        results.pop(key, None)
    print(f"Result files skipped: {skipped}; ambiguous race keys excluded: {len(rejected)}")
    return results


def prediction_key(df):
    columns = ["開催日", "競馬場", "レース番号"]
    if df.empty or df[columns].isna().any().any() or not (df[columns].nunique() == 1).all():
        raise ValueError("Missing or mixed prediction race identity")
    m = re.fullmatch(r"(\d{4})年 第(\d+)回(\d+)日目", str(df["開催日"].iloc[0]))
    race = re.fullmatch(r"(\d+)R", str(df["レース番号"].iloc[0]))
    places = {name: num for num, name in PLACE_NAMES.items()}
    if not m or not race or df["競馬場"].iloc[0] not in places:
        raise ValueError("Invalid prediction race identity")
    key = (int(m[1]), places[df["競馬場"].iloc[0]], int(m[2]), int(m[3]), int(race[1]))
    if not 2000 <= key[0] <= 2099 or min(key[2:]) < 1:
        raise ValueError("Invalid prediction race identity")
    return key


def load_prediction_rows(pred_dir, results):
    rows, seen, rejected = {}, set(), set()
    skipped = 0
    for path in sorted(glob.glob(os.path.join(pred_dir, "pred_*.csv"))):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            key = prediction_key(df)
            if key in seen:
                rejected.add(key)
                raise ValueError("Duplicate prediction race")
            seen.add(key)
            fin = results.get(key)
            if fin is None:
                raise ValueError("No valid result")
            nums = pd.to_numeric(df["馬番"], errors="raise").to_numpy(dtype=float)
            if set(nums) != set(fin):
                raise ValueError("Prediction/result fields differ")
            finishes = np.array([fin[x] for x in nums], dtype=float)
            p1, p2, p3 = [pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
                          for col in ("1着確率", "2着以内確率", "3着以内確率")]
            if not valid_race(nums, finishes, p1, p2, p3):
                raise ValueError("Invalid race probabilities or finishes")
            rows[key] = {
                "key": key, "n": len(df), "nums": nums.astype(int), "fin": finishes,
                "p1": p1 / p1.sum(), "p2": p2, "p3": p3,
            }
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            skipped += 1
    for key in rejected:
        rows.pop(key, None)
    print(f"Prediction files skipped: {skipped}; ambiguous race keys excluded: {len(rejected)}")
    return [rows[key] for key in sorted(rows)]


def combination_records(rows, top_n):
    if top_n < 3:
        raise ValueError("top_n must be at least 3")
    rec = {name: ([], [], []) for name in ("umaren", "wide", "umatan", "fuku3", "tan3")}
    for r in rows:
        p1, p2, p3, fin = r["p1"], r["p2"], r["p3"], r["fin"]
        idx = np.lexsort((r["nums"], -p1))[:top_n].tolist()
        top1, top2, top3 = [int(np.flatnonzero(fin == k)[0]) for k in (1, 2, 3)]
        for i, j in combinations(idx, 2):
            rec["umaren"][0].append(indep_quinella(p2, i, j))
            rec["umaren"][1].append(harville_quinella(p1, i, j))
            rec["umaren"][2].append({i, j} == {top1, top2})
            rec["wide"][0].append(indep_wide(p3, i, j))
            rec["wide"][1].append(harville_wide(p1, i, j))
            rec["wide"][2].append({i, j} <= {top1, top2, top3})
        for i, j in permutations(idx, 2):
            rec["umatan"][0].append(indep_exacta(p1, p2, i, j))
            rec["umatan"][1].append(harville_exacta(p1, i, j))
            rec["umatan"][2].append((i, j) == (top1, top2))
        for a, b, c in combinations(idx, 3):
            rec["fuku3"][0].append(indep_trio(p3, a, b, c))
            rec["fuku3"][1].append(harville_trio(p1, a, b, c))
            rec["fuku3"][2].append({a, b, c} == {top1, top2, top3})
        for a, b, c in permutations(idx, 3):
            rec["tan3"][0].append(indep_trifecta(p1, p2, p3, a, b, c))
            rec["tan3"][1].append(harville_trifecta(p1, a, b, c))
            rec["tan3"][2].append((a, b, c) == (top1, top2, top3))
    return rec


def heldout_frame(rows):
    out = []
    for r in rows:
        for i in range(r["n"]):
            out.append({
                **dict(zip(RACE_COLUMNS, r["key"])),
                "馬番": r["nums"][i], "着順": r["fin"][i],
                "p1": r["p1"][i], "p2": r["p2"][i], "p3": r["p3"][i],
            })
    return pd.DataFrame(out, columns=RACE_COLUMNS + ["馬番", "着順", "p1", "p2", "p3"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-heldout", "--save-oof", dest="save_heldout", action="store_true",
                    help="Export _heldout_probs.csv; --save-oof is a deprecated alias, not OOF")
    ap.add_argument("--top-n-combo", type=int, default=6,
                    help="Evaluate all orders among the top N horses; full-field denominators")
    args = ap.parse_args()
    if args.top_n_combo < 3:
        ap.error("--top-n-combo must be at least 3")
    print("EXPLORATORY ONLY: not adoption evidence; base-model holdout/OOF provenance is unverified.")
    print("Conservative exclusions: incomplete/nonunique top3, any unknown finish, duplicate horses, "
          "mismatched fields, invalid probabilities or ambiguous race keys.")
    base = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base, "predictions")
    results = load_kekka(os.path.join(base, "CSV_kekka"))
    rows = load_prediction_rows(pred_dir, results)
    print(f"Valid matched races: {len(rows)}")
    if args.save_heldout:
        out = heldout_frame(rows)
        out.to_csv(os.path.join(pred_dir, "_heldout_probs.csv"), index=False, encoding="utf-8-sig")
        print(f"Saved predictions/_heldout_probs.csv ({len(out)} rows); not certified OOF.")
    if not rows:
        return
    for k in (1, 2, 3):
        p = np.concatenate([r[f"p{k}"] for r in rows])
        y = np.concatenate([r["fin"] <= k for r in rows])
        print(f"Top{k}: LogLoss={logloss(p, y):.4f} Brier={brier(p, y):.4f} "
              f"mean prediction={p.mean():.4f} actual={y.mean():.4f}")
        if k == 1:
            print(calibration_table(p, y).round(4).to_string())
    print(f"\nTop {args.top_n_combo} horses only; every exacta/trifecta order; "
          "unconditional full-field PL probabilities. Metrics are ticket-weighted, not ROI.")
    print(f"{'type':<8} {'n':>8} {'hit rate':>9} {'ind LL':>9} {'PL LL':>9} {'ind Brier':>10} {'PL Brier':>10}")
    for name, (ind, harv, hit) in combination_records(rows, args.top_n_combo).items():
        if hit:
            print(f"{name:<8} {len(hit):>8} {np.mean(hit):>9.4f} "
                  f"{logloss(ind, hit):>9.4f} {logloss(harv, hit):>9.4f} "
                  f"{brier(ind, hit):>10.4f} {brier(harv, hit):>10.4f}")


if __name__ == "__main__":
    main()
