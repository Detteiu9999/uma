# -*- coding: utf-8 -*-
"""
--min-prob / --min-ev の組み合わせをグリッドサーチし、
回収率が最も高くなる条件を検証する。
check_results.py の判定ロジックを再利用。
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_results import load_results, is_hit, PLACE_NAMES

STAKE = 100

base_dir = os.path.dirname(os.path.abspath(__file__))
suggestions_path = os.path.join(base_dir, "suggestions", "suggested_bets.csv")
kekka_dir = os.path.join(base_dir, "CSV_kekka")

# 結果データの読み込み
results = load_results(kekka_dir)

# 結果データのキー構造（年が含まれる5要素か、含まれない4要素か）を自動判定
is_5_tuple = False
if results:
    sample_key = next(iter(results.keys()))
    if isinstance(sample_key, tuple) and len(sample_key) == 5:
        is_5_tuple = True

# 枠番マップの作成
waku_maps = {}
for path in glob.glob(os.path.join(kekka_dir, "horse_racing_data_*.csv")):
    try:
        kdf = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        continue
        
    if not {"競馬場", "回", "日", "レース", "馬番", "枠番"}.issubset(set(kdf.columns)):
        continue
        
    place_num = int(kdf["競馬場"].iloc[0])
    place = PLACE_NAMES.get(place_num)
    if place is None:
        continue
        
    year = int(kdf["年"].iloc[0]) if "年" in kdf.columns else 0
    kai = int(kdf["回"].iloc[0])
    nichi = int(kdf["日"].iloc[0])
    race = int(kdf["レース"].iloc[0])
    
    # 判定したキー構造に合わせてキーを作成
    if is_5_tuple:
        key = (year, place, kai, nichi, race)
    else:
        key = (place, kai, nichi, race)
        
    waku_maps[key] = {int(r["馬番"]): int(r["枠番"]) for _, r in kdf.iterrows()
                      if str(r["馬番"]).strip() and str(r["枠番"]).strip()}

df = pd.read_csv(suggestions_path, encoding="utf-8-sig")

# 各買い目について 的中/払戻 を事前計算
probs, evs, payouts, bet_types, races = [], [], [], [], []
unmatched = 0
unmatched_samples = []

for _, row in df.iterrows():
    bet_type = str(row["式別"]).strip()
    
    # Excelの数式記法 (="1-14") などになっていた場合のクレンジング
    combo = str(row["買い目"]).strip().replace('="', '').replace('"', '')
    
    try:
        # "20260104_中山1回1日" から年 "2026" を抽出
        year = int(str(row["レースキー"])[:4])
        place = str(row["競馬場"]).strip()
        kai = int(row["回"])
        nichi = int(row["日"])
        race = int(str(row["レース番号"]).replace("R", "").strip())
        
        # 判定したキー構造に合わせてキーを作成
        if is_5_tuple:
            key = (year, place, kai, nichi, race)
        else:
            key = (place, kai, nichi, race)
            
    except (ValueError, TypeError, KeyError):
        unmatched += 1
        continue
        
    finish = results.get(key)
    hit = None
    if finish:
        hit = is_hit(bet_type, combo, finish, waku_maps.get(key, {}))
        
    if hit is None:
        unmatched += 1
        if len(unmatched_samples) < 5:
            unmatched_samples.append(key)
        continue
        
    probs.append(float(row["的中確率(推定)"]))
    evs.append(float(row["期待値"]))
    payouts.append(float(row["オッズ"]) * STAKE if hit else 0.0)
    bet_types.append(bet_type)
    races.append(race)

probs = np.array(probs)
evs = np.array(evs)
payouts = np.array(payouts)
bet_types = np.array(bet_types)
races = np.array(races)
print(f"判定可能: {len(probs)} 点 / 全 {len(df)} 点（未照合 {unmatched} 点）\n")

# ==========================================
# 0件のときのガード節（エラー落ち回避とヒント表示）
# ==========================================
if len(probs) == 0:
    print("エラー: 照合できたデータが 1 件もありません。処理を終了します。")
    print("\n【デバッグ情報: 結合キーがズレている可能性があります】")
    print("▼ 予測ファイル から作った検索キーの例:")
    for k in unmatched_samples:
        print("  ", k)
        
    if results:
        print("\n▼ 結果データ(CSV_kekka) 側に実際に存在しているキーの例:")
        print("  ", list(results.keys())[:5])
    else:
        print("\n▼ 結果データ(CSV_kekka) 自体が空か、正しく読み込めていません。")
    sys.exit(1)


# グリッド定義
prob_grid = [0.1,0.125, 0.15, 0.2, 0.3, 0.5]
ev_grid = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

BET_ORDER = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]

rows = []
type_rows = []
race_rows = []
race_type_rows = []  # レース番号 × 式別 ごとのリスト

for mp in prob_grid:
    for me in ev_grid:
        mask = (probs >= mp) & (evs >= me)
        n = int(mask.sum())
        if n == 0:
            continue
        bet = n * STAKE
        ret = float(payouts[mask].sum())
        hit_n = int((payouts[mask] > 0).sum())
        rows.append({
            "min_prob": mp, "min_ev": me, "点数": n, "的中": hit_n,
            "的中率%": hit_n / n * 100,
            "購入額": bet, "払戻額": int(ret),
            "収支": int(ret) - bet,
            "回収率%": ret / bet * 100,
        })
        
        # 式別ごとの内訳（全レース）
        for bt in BET_ORDER:
            bmask = mask & (bet_types == bt)
            bn = int(bmask.sum())
            if bn == 0:
                continue
            bret = float(payouts[bmask].sum())
            hit_bn = int((payouts[bmask] > 0).sum())
            type_rows.append({
                "min_prob": mp, "min_ev": me, "式別": bt, "点数": bn,
                "的中": hit_bn,
                "的中率%": hit_bn / bn * 100,
                "購入額": bn * STAKE, "払戻額": int(bret),
                "収支": int(bret) - bn * STAKE,
                "回収率%": bret / (bn * STAKE) * 100,
            })
            
        # レース番号ごとの内訳 (1R ～ 12R) 
        for r_num in range(1, 13):
            rmask = mask & (races == r_num)
            rn = int(rmask.sum())
            if rn == 0:
                continue
            rret = float(payouts[rmask].sum())
            hit_rn = int((payouts[rmask] > 0).sum())
            race_rows.append({
                "min_prob": mp, "min_ev": me, "レース番号": r_num, "点数": rn,
                "的中": hit_rn,
                "的中率%": hit_rn / rn * 100,
                "購入額": rn * STAKE, "払戻額": int(rret),
                "収支": int(rret) - rn * STAKE,
                "回収率%": rret / (rn * STAKE) * 100,
            })
            
            # さらに細分化（レース番号 × 式別）
            for bt in BET_ORDER:
                rbmask = rmask & (bet_types == bt)
                rbn = int(rbmask.sum())
                if rbn == 0:
                    continue
                rbret = float(payouts[rbmask].sum())
                hit_rbn = int((payouts[rbmask] > 0).sum())
                race_type_rows.append({
                    "min_prob": mp, "min_ev": me, "レース番号": r_num, "式別": bt,
                    "点数": rbn, "的中": hit_rbn, "的中率%": hit_rbn / rbn * 100,
                    "購入額": rbn * STAKE, "払戻額": int(rbret),
                    "収支": int(rbret) - rbn * STAKE,
                    "回収率%": rbret / (rbn * STAKE) * 100,
                })


res = pd.DataFrame(rows)

if res.empty:
    print("条件を満たすデータがありませんでした。処理を終了します。")
    sys.exit(0)

# CSV 保存: 全体結果
try:
    res.to_csv(os.path.join(base_dir, "grid_search_result.csv"),
               index=False, encoding="utf-8-sig")
except PermissionError:
    print("[警告] grid_search_result.csv が他のプロセスで開かれているため保存をスキップ")

# CSV 保存: 式別集計（全レース）
res_type = pd.DataFrame(type_rows)
try:
    res_type.to_csv(os.path.join(base_dir, "grid_search_by_type.csv"),
                    index=False, encoding="utf-8-sig")
except PermissionError:
    print("[警告] grid_search_by_type.csv が他のプロセスで開かれているため保存をスキップ")

# CSV 保存: レース番号別集計（全式別合算）
res_race = pd.DataFrame(race_rows)
try:
    res_race.to_csv(os.path.join(base_dir, "grid_search_by_race.csv"),
                    index=False, encoding="utf-8-sig")
except PermissionError:
    print("[警告] grid_search_by_race.csv が他のプロセスで開かれているため保存をスキップ")

# CSV 保存: レース番号ごと ＆ 式別 ごとにファイルを分割して出力 (1R～12R)
res_race_type = pd.DataFrame(race_type_rows)
if not res_race_type.empty:
    for r_num in range(1, 13):
        sub_df = res_race_type[res_race_type["レース番号"] == r_num]
        if sub_df.empty:
            continue
            
        file_name = f"grid_search_by_type_{r_num}R.csv"
        try:
            sub_df.to_csv(os.path.join(base_dir, file_name),
                          index=False, encoding="utf-8-sig")
        except PermissionError:
            print(f"[警告] {file_name} が開かれているため保存をスキップ")


# === コンソール出力部 ===
pivot = res.pivot_table(index="min_prob", columns="min_ev", values="回収率%")
print("=== 回収率(%) マトリクス（行: min-prob, 列: min-ev） ===")
print(pivot.round(1).to_string())

pivot_n = res.pivot_table(index="min_prob", columns="min_ev", values="点数")
print("\n=== 点数マトリクス ===")
print(pivot_n.fillna(0).astype(int).to_string())

print("\n=== 回収率トップ10（全条件） ===")
print(res.sort_values("回収率%", ascending=False).head(10).to_string(index=False))

print("\n=== 回収率トップ10（購入点数 30点以上に限定） ===")
print(res[res["点数"] >= 30].sort_values("回収率%", ascending=False).head(10).to_string(index=False))

print("\n=== 収支トップ10（購入点数 30点以上に限定） ===")
print(res[res["点数"] >= 30].sort_values("収支", ascending=False).head(10).to_string(index=False))

# ============================================================
# 式別ごとの回収率マトリクス（全レース合算）
# ============================================================
print("\n" + "=" * 70)
print("=== 式別ごとの回収率(%) マトリクス（行: min-prob, 列: min-ev） ===")
print("=" * 70)
for bt in BET_ORDER:
    sub = res_type[res_type["式別"] == bt]
    if sub.empty:
        continue
    pivot_bt = sub.pivot_table(index="min_prob", columns="min_ev",
                               values="回収率%")
    pivot_n_bt = sub.pivot_table(index="min_prob", columns="min_ev",
                                 values="点数")
    print(f"\n■ {bt}（回収率% / 括弧内は点数）")
    for mp in pivot_bt.index:
        cells = []
        for me in pivot_bt.columns:
            r_val = pivot_bt.loc[mp, me]
            nn = pivot_n_bt.loc[mp, me] if me in pivot_n_bt.columns else np.nan
            if pd.isna(r_val) or pd.isna(nn):
                cells.append("-")
            else:
                cells.append(f"{r_val:.0f}({int(nn)})")
        print(f"  prob>={mp:<5} " + "  ".join(f"{c:>10}" for c in cells))
    print(f"  {'':<9} " + "  ".join(f"ev>={me:<7}" for me in pivot_bt.columns))

# ============================================================
# レース番号ごとの回収率マトリクス（全式別合算）
# ============================================================
print("\n" + "=" * 70)
print("=== レース番号ごとの回収率(%) マトリクス（行: min-prob, 列: min-ev） ===")
print("=" * 70)
for r_num in range(1, 13):
    sub = res_race[res_race["レース番号"] == r_num]
    if sub.empty:
        continue
    pivot_r = sub.pivot_table(index="min_prob", columns="min_ev", values="回収率%")
    pivot_n_r = sub.pivot_table(index="min_prob", columns="min_ev", values="点数")
    
    print(f"\n■ {r_num}R（回収率% / 括弧内は点数）")
    for mp in pivot_r.index:
        cells = []
        for me in pivot_r.columns:
            r_val = pivot_r.loc[mp, me]
            nn = pivot_n_r.loc[mp, me] if me in pivot_n_r.columns else np.nan
            if pd.isna(r_val) or pd.isna(nn):
                cells.append("-")
            else:
                cells.append(f"{r_val:.0f}({int(nn)})")
        print(f"  prob>={mp:<5} " + "  ".join(f"{c:>10}" for c in cells))
    print(f"  {'':<9} " + "  ".join(f"ev>={me:<7}" for me in pivot_r.columns))

print(f"\n※ レース番号別の式別集計ファイル（grid_search_by_type_1R.csv など）も {base_dir} に出力しました。")