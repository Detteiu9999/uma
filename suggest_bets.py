# -*- coding: utf-8 -*-
"""
suggest_bets.py
predictions/ の予測確率（1着/2着以内/3着以内）と odds/ の最新オッズから、
期待値（= 的中確率 × オッズ）の高い買い目を計算してCSVに出力する単独スクリプト。

フィルタルール（安定化のため）:
  各馬について、
    ・1着確率   < 0.1 → その馬が「1着指定」される買い目（単勝・馬単の1着・3連単の1着）を除外
    ・2着以内確率 < 0.1 → その馬が「2着以内指定」される買い目（馬連・ワイド・馬単・3連複・3連単の2着）を除外
    ・3着以内確率 < 0.1 → その馬が「3着以内指定」される買い目（複勝・ワイド・3連複・3連単の3着）を除外
  例: 2着以内/3着以内が 0.1 未満でも 1着確率が 0.2 なら単勝は候補に残る。

確率の計算:
  ・単勝/複勝         : その馬の 1着確率 / 3着以内確率
  ・枠連/馬連/馬単/3連複/3連単/ワイド:
      各馬の 1着確率を強さとした Plackett–Luce モデルによる推定。
      枠連は全出走馬の馬単確率を、対象の枠組合せについて合算する。
      実際の的中確率や独立性近似に対する優位性を保証するものではない。
    ※--indep を付けると従来の独立性近似（単純な掛け算）に戻せます。

使い方:
    python suggest_bets.py                          # 全レースを処理してCSV出力
    python suggest_bets.py --place 札幌 --race 11
    python suggest_bets.py --min-ev 1.2             # 期待値1.2以上のものを出力
"""

import argparse
import glob
import math
import os
import re
from itertools import permutations
from numbers import Integral

import numpy as np
import pandas as pd

# ============================================================
# 設定
# ============================================================

PROB_THRESHOLD = 0.1     # この確率を下回る馬は関連買い目から除外
DEFAULT_MIN_EV = 1.0     # 期待値の下限（これ以上を提案）

DISCORD_BET_RULES = {
    "馬連": (0.15, 1.0, math.inf),
    "馬単": (0.1, 1.0, 1.5),
    "3連複": (0.125, 1.5, 3.0),
    "ワイド": (0.3, 1.2, math.inf),
    "複勝": (0.5, 1.2, math.inf),
}

# 競馬場コード → 場名
PLACE_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}


# ============================================================
# データ読み込み
# ============================================================

def race_key_from_filename(path):
    """odds_20260906_札幌2回6日_11R_tanpuku.csv -> ('20260906_札幌2回6日', 11)"""
    m = re.search(r"odds_(\d{8})_([^_]+)_(\d+)R_", os.path.basename(path))
    if not m:
        return None
    return (f"{m.group(1)}_{m.group(2)}", int(m.group(3)))


def load_predictions(pred_dir):
    """pred_*.csv を読み、(競馬場, 回, 日, レース番号) -> DataFrame の辞書を返す。"""
    races = {}
    for path in glob.glob(os.path.join(pred_dir, "pred_*.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        need = {"開催日", "競馬場", "レース番号", "馬番", "馬名",
                "1着確率", "2着以内確率", "3着以内確率"}
        if not need <= set(df.columns):
            continue
        m = re.match(r"^(\d{4})年 第(\d+)回(\d+)日目$", str(df["開催日"].iloc[0]))
        if not m:
            continue
        key = (str(df["競馬場"].iloc[0]), int(m.group(2)), int(m.group(3)),
               int(str(df["レース番号"].iloc[0]).replace("R", "")))
        races[key] = df
    return races


def load_odds(odds_dir):
    """odds/*.csv を読み、(race_key, race_num, 式別キー) -> DataFrame の辞書を返す。"""
    odds = {}
    for path in glob.glob(os.path.join(odds_dir, "odds_*.csv")):
        rk = race_key_from_filename(path)
        if not rk:
            continue
        bet_key = os.path.basename(path).rsplit("_", 1)[-1].replace(".csv", "")
        try:
            odds[(rk[0], rk[1], bet_key)] = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
    return odds


# ============================================================
# 買い目生成
# ============================================================

class Horse:
    def __init__(self, row):
        self.num = int(row["馬番"])
        self.name = str(row["馬名"])
        self.p1 = float(row["1着確率"])
        self.p2 = float(row["2着以内確率"])
        self.p3 = float(row["3着以内確率"])
        waku = row.get("枠番", "")
        self.waku = int(waku) if str(waku) not in ("", "nan") else None

    @property
    def ok_win(self):
        return self.p1 >= PROB_THRESHOLD

    @property
    def ok_place2(self):
        return self.p2 >= PROB_THRESHOLD

    @property
    def ok_place3(self):
        return self.p3 >= PROB_THRESHOLD


def _validate_probabilities(p, normalized=True):
    try:
        p = np.asarray(p, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be a finite numeric vector") from exc
    if (p.ndim != 1 or not len(p) or not np.all(np.isfinite(p))
            or np.any(p < 0) or np.any(p > 1)):
        raise ValueError("Probabilities must be a nonempty vector in [0, 1]")
    total = math.fsum(p)
    if total <= 0:
        raise ValueError("Probabilities must have positive total mass")
    if normalized and not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Harville probabilities must sum to one")
    return p


def _validate_indices(p, indices):
    if any(not isinstance(i, Integral) or isinstance(i, (bool, np.bool_))
           or i < 0 or i >= len(p) for i in indices):
        raise ValueError("Horse indices must be integers within the field")
    if len(set(indices)) != len(indices):
        raise ValueError("Horse indices must be distinct")


def _ordered_probability(p, indices):
    remaining = list(range(len(p)))
    prob = 1.0
    for i in indices:
        denominator = math.fsum(p[j] for j in remaining)
        if denominator == 0 or p[i] == 0:
            return 0.0
        prob *= p[i] / denominator
        remaining.remove(i)
    return prob


def harville_exacta(p, a, b):
    p = _validate_probabilities(p)
    _validate_indices(p, (a, b))
    return _ordered_probability(p, (a, b))


def harville_quinella(p, a, b):
    p = _validate_probabilities(p)
    _validate_indices(p, (a, b))
    return math.fsum(_ordered_probability(p, order)
                     for order in permutations((a, b)))


def harville_trifecta(p, a, b, c):
    p = _validate_probabilities(p)
    _validate_indices(p, (a, b, c))
    return _ordered_probability(p, (a, b, c))


def harville_trio(p, a, b, c):
    p = _validate_probabilities(p)
    _validate_indices(p, (a, b, c))
    return math.fsum(_ordered_probability(p, order)
                     for order in permutations((a, b, c)))


def harville_wide(p, a, b):
    p = _validate_probabilities(p)
    _validate_indices(p, (a, b))
    if len(p) == 2:
        return harville_quinella(p, a, b)
    return min(1.0, math.fsum(
        _ordered_probability(p, order)
        for x in range(len(p)) if x not in (a, b)
        for order in permutations((a, b, x))))


def harville_wakuren(p, frames, w1, w2):
    p = _validate_probabilities(p)
    if len(frames) != len(p):
        raise ValueError("Each horse must have a frame entry")
    return math.fsum(
        _ordered_probability(p, (a, b))
        for a in range(len(p)) for b in range(len(p))
        if a != b and ((frames[a] == w1 and frames[b] == w2)
                       or (frames[a] == w2 and frames[b] == w1)))


def win_probs_vector(horses):
    p = _validate_probabilities([h.p1 for h in horses], normalized=False)
    p = p / max(p)
    return p / math.fsum(p)


def parse_wide_odds(text):
    """'27.8-29.9' -> (27.8, 29.9)。パースできなければ (None, None)"""
    m = re.match(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$", str(text))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def meets_discord_rules(bet_type, prob, ev):
    rule = DISCORD_BET_RULES.get(bet_type)
    return (rule is not None and math.isfinite(prob) and math.isfinite(ev)
            and rule[0] <= prob <= 1.0 and rule[1] <= ev <= rule[2])


def suggest_for_race(horses, odds_for_race, min_ev, use_harville=True,
                     discord_only=False):
    by_num = {h.num: h for h in horses}
    if len(by_num) != len(horses):
        raise ValueError("Horse numbers must be distinct")
    idx_of = {h.num: i for i, h in enumerate(horses)}
    pwin = win_probs_vector(horses)
    suggestions = []

    def add(bet_type, combo, prob, odds_val, note=""):
        try:
            prob = float(prob)
            odds_f = float(odds_val)
        except (TypeError, ValueError, OverflowError):
            return
        if (not math.isfinite(prob) or not math.isfinite(odds_f)
                or not PROB_THRESHOLD <= prob <= 1.0 or odds_f <= 0):
            return

        ev = prob * odds_f
        if discord_only and not meets_discord_rules(bet_type, prob, ev):
            return
        if math.isfinite(ev) and ev >= min_ev:
            suggestions.append({
                "式別": bet_type, "買い目": combo,
                "的中確率(推定)": round(prob, 4),
                "オッズ": odds_f,
                "期待値": round(ev, 3),
                "備考": note,
            })

    # --- 単勝・複勝 ---
    df = odds_for_race.get("tanpuku")
    if df is not None:
        for _, row in df.iterrows():
            try:
                num = int(row["馬番"])
            except (TypeError, ValueError):
                continue
            h = by_num.get(num)
            if not h:
                continue
            if h.ok_win:
                add("単勝", f"{num} {h.name}", h.p1, row.get("単勝オッズ"))
            if h.ok_place3:
                # 複勝オッズは幅があるため下限（堅め）で評価
                add("複勝", f"{num} {h.name}", h.p3, row.get("複勝オッズ下限"),
                    note="複勝オッズは下限値で評価")

    # --- 枠連 ---
    df = odds_for_race.get("wakuren")
    if df is not None:
        frames = [h.waku for h in horses]
        waku_p2 = {}
        for h in horses:
            if h.waku is not None and h.ok_place2:
                waku_p2[h.waku] = waku_p2.get(h.waku, 0.0) + h.p2
        for _, row in df.iterrows():
            try:
                w1, w2 = int(row["枠番1"]), int(row["枠番2"])
            except (TypeError, ValueError):
                continue
            if use_harville:
                prob = harville_wakuren(pwin, frames, w1, w2)
                add("枠連", f"枠{w1}-枠{w2}", prob, row.get("枠連オッズ"))
            else:
                if w1 in waku_p2 and w2 in waku_p2:
                    add("枠連", f"枠{w1}-枠{w2}", waku_p2[w1] * waku_p2[w2],
                        row.get("枠連オッズ"))

    # --- 馬連・ワイド ---
    for bet_key, bet_name, prob_attr, ok_attr in [
        ("umaren", "馬連", "p2", "ok_place2"),
        ("wide", "ワイド", "p3", "ok_place3"),
    ]:
        df = odds_for_race.get(bet_key)
        if df is None:
            continue
        for _, row in df.iterrows():
            try:
                a, b = int(row["馬番1"]), int(row["馬番2"])
            except (TypeError, ValueError):
                continue
            ha, hb = by_num.get(a), by_num.get(b)
            if not ha or not hb or a == b:
                continue
            if not (getattr(ha, ok_attr) and getattr(hb, ok_attr)):
                continue
            if use_harville:
                ia, ib = idx_of[a], idx_of[b]
                prob = (harville_quinella(pwin, ia, ib) if bet_key == "umaren"
                        else harville_wide(pwin, ia, ib))
            else:
                prob = getattr(ha, prob_attr) * getattr(hb, prob_attr)
            if bet_key == "wide":
                lo, hi = parse_wide_odds(row.get("ワイドオッズ"))
                add(bet_name, f"{a}-{b}", prob, lo, note="ワイドオッズは下限値で評価")
            else:
                add(bet_name, f"{a}-{b}", prob, row.get(f"{bet_name}オッズ"))

    # --- 馬単 ---
    df = odds_for_race.get("umatan")
    if df is not None:
        for _, row in df.iterrows():
            try:
                a, b = int(row["馬番1"]), int(row["馬番2"])
            except (TypeError, ValueError):
                continue
            ha, hb = by_num.get(a), by_num.get(b)
            if not ha or not hb or a == b:
                continue
            # 1着指定の馬は ok_win、2着指定の馬は ok_place2 が必要
            if not (ha.ok_win and hb.ok_place2):
                continue
            if use_harville:
                prob = harville_exacta(pwin, idx_of[a], idx_of[b])
            else:
                prob = ha.p1 * hb.p2
            add("馬単", f"{a}→{b}", prob, row.get("馬単オッズ"))

    # --- 3連複 ---
    df = odds_for_race.get("fuku3")
    if df is not None:
        for _, row in df.iterrows():
            try:
                a, b, c = int(row["馬番1"]), int(row["馬番2"]), int(row["馬番3"])
            except (TypeError, ValueError):
                continue
            hs = [by_num.get(x) for x in (a, b, c)]
            if len({a, b, c}) != 3 or any(h is None for h in hs):
                continue
            if not all(h.ok_place3 for h in hs):
                continue
            if use_harville:
                prob = harville_trio(pwin, idx_of[a], idx_of[b], idx_of[c])
            else:
                prob = hs[0].p3 * hs[1].p3 * hs[2].p3
            add("3連複", f"{a}-{b}-{c}", prob, row.get("3連複オッズ"))

    # --- 3連単 ---
    df = odds_for_race.get("tan3")
    if df is not None:
        for _, row in df.iterrows():
            try:
                a, b, c = int(row["1着馬番"]), int(row["2着馬番"]), int(row["3着馬番"])
            except (TypeError, ValueError):
                continue
            ha, hb, hc = by_num.get(a), by_num.get(b), by_num.get(c)
            if len({a, b, c}) != 3 or not ha or not hb or not hc:
                continue
            if not (ha.ok_win and hb.ok_place2 and hc.ok_place3):
                continue
            if use_harville:
                prob = harville_trifecta(pwin, idx_of[a], idx_of[b], idx_of[c])
            else:
                prob = ha.p1 * hb.p2 * hc.p3
            add("3連単", f"{a}→{b}→{c}", prob, row.get("3連単オッズ"))

    return suggestions


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="予測確率と最新オッズから期待値の高い買い目を計算しCSV出力する")
    parser.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV,
                        help=f"期待値の下限（デフォルト {DEFAULT_MIN_EV}）")
    parser.add_argument("--place", help="競馬場名で絞り込み（例: 札幌）")
    parser.add_argument("--place-code", type=int,
                        help="競馬場コードで絞り込み（1=札幌 … 10=小倉）")
    parser.add_argument("--race", type=int, help="レース番号で絞り込み")
    parser.add_argument("--indep", action="store_true",
                        help="組合せ確率を従来の独立性近似（掛け算）で計算する")
    parser.add_argument("--output", help="出力CSVのパス")
    args = parser.parse_args()

    place_filter = args.place
    if args.place_code:
        place_filter = PLACE_NAMES.get(args.place_code)
        if place_filter is None:
            print(f"[エラー] --place-code は 1〜10 で指定してください: {args.place_code}")
            return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, "predictions")
    odds_dir = os.path.join(base_dir, "odds")
    out_dir = os.path.join(base_dir, "suggestions")

    predictions = load_predictions(pred_dir)
    odds = load_odds(odds_dir)
    if not predictions:
        print("[エラー] predictions/ に予測CSVが見つかりません。先に predict_model.py を実行してください。")
        return
    if not odds:
        print("[エラー] odds/ にオッズCSVが見つかりません。先に fetch_odds.py を実行してください。")
        return

    # odds 側の race_key -> (場名, 回, 日) を復元
    # 例: '20260906_札幌2回6日' -> ('札幌', 2, 6)
    odds_race_info = {}
    for (race_key, race_num, _bet) in odds:
        if race_key in odds_race_info:
            continue
        m = re.match(r"^(\d{8})_(.+?)(\d+)回(\d+)日$", race_key)
        if m:
            odds_race_info[race_key] = (m.group(2), int(m.group(3)), int(m.group(4)))

    all_rows = []
    for race_key, (place, kai, day) in sorted(odds_race_info.items()):
        if place_filter and place != place_filter:
            continue
        race_nums = sorted({rn for (rk, rn, _b) in odds if rk == race_key})
        for race_num in race_nums:
            if args.race and race_num != args.race:
                continue
            pred_key = (place, kai, day, race_num)
            df_pred = predictions.get(pred_key)
            if df_pred is None:
                continue
            odds_for_race = {bet: df for (rk, rn, bet), df in odds.items()
                             if rk == race_key and rn == race_num}
            horses = [Horse(row) for _, row in df_pred.iterrows()]
            suggestions = suggest_for_race(horses, odds_for_race, args.min_ev,
                                           use_harville=not args.indep)
            if not suggestions:
                continue

            suggestions.sort(key=lambda x: -x["期待値"])

            for s in suggestions:
                all_rows.append({
                    "レースキー": race_key, "競馬場": place, "回": kai, "日": day,
                    "レース番号": f"{race_num}R", **s,
                })

    if not all_rows:
        print("条件に合う買い目は見つかりませんでした。")
        print("（--min-ev を下げるか、対象レースの予測CSV/オッズCSVがあるか確認してください）")
        return

    out_path = args.output or os.path.join(out_dir, "suggested_bets.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    df_out = pd.DataFrame(all_rows)
    if "買い目" in df_out.columns:
        df_out["買い目"] = df_out["買い目"].apply(lambda x: f'="{x}"')

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"全候補を保存しました: {out_path} ({len(all_rows)}件)")


if __name__ == "__main__":
    main()