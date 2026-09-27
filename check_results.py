# -*- coding: utf-8 -*-
"""
check_results.py
suggestions/suggested_bets.csv の買い目のうち、
「的中確率(推定) >= 0.1 かつ 期待値 >= 2.0」に絞って購入したと仮定し、
CSV_kekka/ のレース結果と照合して最終成績（収支・的中率・回収率）を表示する。

前提:
  - 1点あたり 100 円購入として計算
  - 払戻は「購入時オッズ × 100 円」（JRAの実払戻ではなく取得オッズベースの概算）
  - 複勝・ワイドはオッズが幅（下限-上限）のため、下限値で評価（堅めの概算）

使い方:
    python check_results.py
    python check_results.py --min-prob 0.1 --min-ev 2.0 --stake 100
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd

# ============================================================
# 設定
# ============================================================

BET_ORDER = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]

# 競馬場コード → 場名
PLACE_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}


# ============================================================
# レース結果の読み込み
# ============================================================

def parse_finish(text):
    """'1着' -> 1, '中止' など -> None"""
    m = re.match(r"^\s*(\d+)着", str(text))
    return int(m.group(1)) if m else None


def load_results(kekka_dir):
    """
    CSV_kekka/horse_racing_data_*.csv を読み、
    (場名, 回, 日, レース番号) -> {馬番: 着順} の辞書を返す。
    """
    results = {}
    for path in glob.glob(os.path.join(kekka_dir, "horse_racing_data_*.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if not {"競馬場", "回", "日", "レース", "馬番", "着順"} <= set(df.columns):
            print(f"[警告] カラム不足でスキップ: {path} → {list(df.columns)}")
            continue
        place = PLACE_NAMES.get(int(df["競馬場"].iloc[0]))
        if place is None:
            continue
        key = (place, int(df["回"].iloc[0]), int(df["日"].iloc[0]), int(df["レース"].iloc[0]))
        finish = {}
        for _, row in df.iterrows():
            try:
                num = int(row["馬番"])
            except (TypeError, ValueError):
                continue
            f = parse_finish(row["着順"])
            if f is not None:
                finish[num] = f
        results[key] = finish
    return results


# ============================================================
# 買い目の判定
# ============================================================

def parse_combo(bet_type, combo):
    if not isinstance(combo, str):
        return None
    combo = combo.strip()
    if combo.startswith('="') and combo.endswith('"'):
        combo = combo[2:-1].replace('""', '"').strip()
    if bet_type in ("単勝", "複勝"):
        m = re.match(r"^(\d+)", combo)
        return (int(m.group(1)),) if m else None
    if bet_type == "枠連":
        m = re.match(r"^枠(\d+)-枠(\d+)$", combo)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if bet_type in ("馬連", "ワイド"):
        m = re.match(r"^(\d+)-(\d+)$", combo)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if bet_type == "馬単":
        m = re.match(r"^(\d+)→(\d+)$", combo)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if bet_type == "3連複":
        m = re.match(r"^(\d+)-(\d+)-(\d+)$", combo)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    if bet_type == "3連単":
        m = re.match(r"^(\d+)→(\d+)→(\d+)$", combo)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    return None


def is_hit(bet_type, combo, finish, waku_map, starters=None):
    """
    買い目が的中したか判定する。
    finish: {馬番: 着順}  waku_map: {馬番: 枠番}
    starters: 出走頭数（複勝の的中条件が7頭以下で2着までに変わるため。
              省略時は着順が取れた頭数で代用）
    """
    nums = parse_combo(bet_type, combo)
    if nums is None:
        return None  # 判定不能

    # 着順が取れている馬だけで順位リストを作る
    ranked = sorted(finish.items(), key=lambda x: x[1])
    if len(ranked) < 3:
        return None
    top1 = ranked[0][0]
    top2 = ranked[1][0]
    top3 = ranked[2][0]

    if bet_type == "単勝":
        return nums[0] == top1
    if bet_type == "複勝":
        # 出走7頭以下のレースでは2着までが払い戻し対象
        field = starters if starters is not None else len(finish)
        return finish.get(nums[0], 99) <= (2 if field <= 7 else 3)
    if bet_type == "枠連":
        w1, w2 = waku_map.get(top1), waku_map.get(top2)
        if w1 is None or w2 is None:
            return None
        return frozenset(nums) == frozenset((w1, w2))
    if bet_type == "馬連":
        return frozenset(nums) == frozenset((top1, top2))
    if bet_type == "ワイド":
        s = frozenset(nums)
        return s <= frozenset((top1, top2, top3)) and len(s) == 2
    if bet_type == "馬単":
        return nums == (top1, top2)
    if bet_type == "3連複":
        return frozenset(nums) == frozenset((top1, top2, top3))
    if bet_type == "3連単":
        return nums == (top1, top2, top3)
    return None


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="suggested_bets.csv の買い目を CSV_kekka/ の結果と照合して成績を表示する")
    parser.add_argument("--min-prob", type=float, default=0.1,
                        help="購入対象とする的中確率(推定)の下限（デフォルト 0.1）")
    parser.add_argument("--min-ev", type=float, default=2.0,
                        help="購入対象とする期待値の下限（デフォルト 2.0）")
    parser.add_argument("--stake", type=int, default=100,
                        help="1点あたりの購入金額（デフォルト 100円）")
    parser.add_argument("--detail", action="store_true",
                        help="的中した買い目の内訳をすべて表示する")
    parser.add_argument("--race", type=int, help="レース番号で絞り込み")
    parser.add_argument("--place", help="競馬場名で絞り込み")
    parser.add_argument("--suggestions", help="照合する買い目CSVのパス")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    suggestions_path = args.suggestions or os.path.join(
        base_dir, "suggestions", "suggested_bets.csv")
    kekka_dir = os.path.join(base_dir, "CSV_kekka")

    if not os.path.exists(suggestions_path):
        print(f"[エラー] {suggestions_path} が見つかりません。先に suggest_bets.py を実行してください。")
        return
    results = load_results(kekka_dir)
    if not results:
        print(f"[エラー] {kekka_dir} にレース結果CSVが見つかりません。")
        return

    df = pd.read_csv(suggestions_path, encoding="utf-8-sig")
    target = df[(df["的中確率(推定)"] >= args.min_prob) & (df["期待値"] >= args.min_ev)]
    if args.place:
        target = target[target["競馬場"] == args.place]
    if args.race is not None:
        race_numbers = pd.to_numeric(
            target["レース番号"].astype(str).str.strip().str.removesuffix("R"),
            errors="coerce")
        target = target[race_numbers == args.race]
    print(f"=== 購入シミュレーション ===")
    print(f"条件: 的中確率(推定) >= {args.min_prob} かつ 期待値 >= {args.min_ev} / 1点 {args.stake} 円")
    print(f"対象買い目: {len(target)} 点（全 {len(df)} 点中）")
    print()

    # 枠番マップ（枠連判定用）: 結果CSVの枠番を使う
    # 出走頭数（複勝判定用）: 結果CSVの行数（中止馬も出走頭数に含める）
    waku_maps = {}
    starters = {}
    for path in glob.glob(os.path.join(kekka_dir, "horse_racing_data_*.csv")):
        try:
            kdf = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if not {"競馬場", "回", "日", "レース", "馬番", "枠番"} <= set(kdf.columns):
            continue
        place = PLACE_NAMES.get(int(kdf["競馬場"].iloc[0]))
        if place is None:
            continue
        key = (place, int(kdf["回"].iloc[0]), int(kdf["日"].iloc[0]), int(kdf["レース"].iloc[0]))
        waku_maps[key] = {int(r["馬番"]): int(r["枠番"]) for _, r in kdf.iterrows()
                          if str(r["馬番"]).strip() and str(r["枠番"]).strip()}
        starters[key] = len(kdf)

    total_bet = 0
    total_return = 0.0
    hit_count = 0
    unknown = 0
    by_type = defaultdict(lambda: {"count": 0, "hit": 0, "bet": 0, "return": 0.0})
    hit_details = []

    for _, row in target.iterrows():
        bet_type = str(row["式別"])
        combo = str(row["買い目"])
        key = (str(row["競馬場"]), int(row["回"]), int(row["日"]),
               int(str(row["レース番号"]).replace("R", "")))
        finish = results.get(key)
        if not finish:
            unknown += 1
            print(f"[未照合] {key} 式別={bet_type} 買い目={combo}")
            continue
        hit = is_hit(bet_type, combo, finish, waku_maps.get(key, {}),
                     starters.get(key))
        if hit is None:
            unknown += 1
            continue

        total_bet += args.stake
        by_type[bet_type]["count"] += 1
        by_type[bet_type]["bet"] += args.stake
        if hit:
            payout = float(row["オッズ"]) * args.stake
            total_return += payout
            hit_count += 1
            by_type[bet_type]["hit"] += 1
            by_type[bet_type]["return"] += payout
            hit_details.append({
                "レース": f"{key[0]}{key[1]}回{key[2]}日 {key[3]}R",
                "式別": bet_type, "買い目": combo,
                "オッズ": row["オッズ"], "払戻": int(payout),
            })

    print("----------------------------------------")
    print(f"購入点数 : {sum(v['count'] for v in by_type.values())} 点"
          f"（結果未照合 {unknown} 点）")
    print(f"的中点数 : {hit_count} 点")
    judged = sum(v['count'] for v in by_type.values())
    if judged:
        print(f"的中率   : {hit_count / judged * 100:.1f} %")
    print(f"購入金額 : {total_bet:,} 円")
    print(f"払戻金額 : {int(total_return):,} 円")
    print(f"収支     : {int(total_return) - total_bet:+,} 円")
    if total_bet:
        print(f"回収率   : {total_return / total_bet * 100:.1f} %")
    print("----------------------------------------")

    print("\n■ 式別ごとの成績")
    print(f"{'式別':<6} {'点数':>4} {'的中':>4} {'的中率':>8} {'購入額':>8} {'払戻額':>8} {'収支':>9} {'回収率':>7}")
    for bet_type in BET_ORDER:
        v = by_type.get(bet_type)
        if not v or v["count"] == 0:
            continue
        hit_rate = v["hit"] / v["count"] * 100
        recovery = v["return"] / v["bet"] * 100 if v["bet"] else 0
        profit = int(v["return"]) - v["bet"]
        print(f"{bet_type:<6} {v['count']:>4} {v['hit']:>4} {hit_rate:>7.1f}% "
              f"{v['bet']:>7,}円 {int(v['return']):>7,}円 {profit:>+8,}円 {recovery:>6.1f}%")

    if hit_details:
        print(f"\n■ 的中した買い目（{len(hit_details)} 点）")
        limit = len(hit_details) if args.detail else 20
        for d in hit_details[:limit]:
            print(f"  {d['レース']:<14} {d['式別']:<4} {d['買い目']:<12} "
                  f"オッズ={d['オッズ']:>8} 払戻={d['払戻']:>8,}円")
        if not args.detail and len(hit_details) > limit:
            print(f"  ... ほか {len(hit_details) - limit} 点（--detail で全表示）")


if __name__ == "__main__":
    main()
