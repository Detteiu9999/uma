# -*- coding: utf-8 -*-
"""
競馬 着順予測 機械学習モデル（ランキング学習版）
=====================================================
CSV_past（過去データ）から学習し、CSV_predict（予測対象レース）について
出走各馬ごとに
    ・1着になる確率        (win)
    ・2着以内に入る確率    (top2)
    ・3着以内に入る確率    (top3)
を算出し、それぞれ確率の高い順に表示・保存する。

モデル:
    XGBoost の LambdaRank（rank:pairwise）によるレース内ランキング学習。
    ラベルは「頭数 − 着順」（着順が良いほど大きい）で、着順全体の情報を使う。
    得られたスコアを Plackett–Luce モデルで確率化する:
        win  = レース内 softmax（合計 = 1）
        top2 / top3 = PL モデルの上位 k 入り確率の厳密解（合計 = 2 / 3）
    softmax の温度は時系列検証データで 1着 LogLoss 最小になるよう較正する。

カテゴリ特徴（競馬場・性別・回り）:
    XGBoost のネイティブカテゴリサポート（enable_categorical=True）を使用。
    騎手・調教師は高カーディナリティのため Target Encoding（OOF）で数値化。

通算成績（脚質:逃先差追 / 通算_*）のスナップショット時点:
    学習時に自動診断する。
        pre  : レース前時点の値       → そのまま使用
        post : レース直後の値         → 当該レース分（今回距離が該当する距離帯列のみ）を差し引く
        leak : 後日まとめて取得した値 → 未来の成績を含むため通算系特徴を自動で除外
    --drop-career で強制的に除外することもできる。

ハイパーパラメータ:
    --tune で Optuna 探索。結果は models/tuned_params.json に保存され自動適用される。

使い方:
    python predict_model.py                 # 学習 → 予測
    python predict_model.py --train         # 学習のみ
    python predict_model.py --predict-only  # 保存済みモデルで予測のみ
    python predict_model.py --tune          # パラメータ自動探索（Optuna）
    python predict_model.py --tune --trials 100
    python predict_model.py --train --drop-career   # 通算系特徴を使わずに学習
出力:
    - 標準出力に各レースのランキング
    - predictions/ ディレクトリに各レースの予測CSV
    - predictions/_all_predictions.csv に全レース結合版

要件: xgboost >= 2.0（rank:pairwise の lambdarank_* パラメータ）
"""

import os
import re
import glob
import json
import argparse
import warnings
import unicodedata
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("xgboost が必要です:  python -m pip install xgboost")

from sklearn.model_selection import GroupKFold

# ============================================================
# パス設定
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PAST_DIR = os.path.join(BASE_DIR, "CSV_past")
PRED_DIR = os.path.join(BASE_DIR, "CSV_predict")
OUT_DIR = os.path.join(BASE_DIR, "predictions")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = "model_rank.json"           # ランキングモデル (XGBoost)
LGB_MODEL_FILE = "model_rank_lgb.txt"    # ランキングモデル (LightGBM, アンサンブル用)
MT_MODEL_FILE = "model_mt_win.json"
META_FILE = "meta.json"                  # 特徴列・カテゴリマップ・温度・学習情報
TUNED_PARAMS_FILE = "tuned_params.json"  # Optuna で探索した最適パラメータ
META_VERSION = 8

# 競馬場番号 → 競馬場名（JRA標準割当）
PLACE_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}
PLACE_NAMES_F = {float(k): v for k, v in PLACE_NAMES.items()}

# 過去走の競馬場短縮表記 → 競馬場番号
PLACE_ABBR = {
    "札": 1, "函": 2, "福": 3, "新": 4, "東": 5,
    "中": 6, "名": 7, "京": 8, "阪": 9, "小": 10,
}

# ============================================================
# 列定義
# ============================================================
PAST_PREFIXES = ["前走", "2走前", "3走前", "4走前", "5走前"]

PAST_NUMERIC_FIELDS = [
    "スピード指数", "先行指数", "ペース指数", "上がり指数",
    "タイム差", "着順", "人気", "頭数", "馬番", "距離", "斤量",
    "体重増減", "上がり3F順位", "上3F", "馬体重",
]

BASE_NUM_FIELDS = ["枠番", "馬番", "年齢", "距離"]

CAT_SUFFIX = "__cat"
MIN_CAT_COUNT = 30

STYLE_COL = "脚質:逃先差追"
CAREER_CANON = ["通算_全", "通算_距離帯A", "通算_距離帯B", "通算_距離帯C"]

# 距離帯列の範囲。normalize_career_columns が元ヘッダから解析できた場合はそれを優先し、
# 解析できない場合はこのデフォルトを使う（データの定義に合わせて調整してください）。
CAREER_DIST_BANDS_DEFAULT = {
    "通算_距離帯A": (0, 1400),
    "通算_距離帯B": (1401, 2000),
    "通算_距離帯C": (2001, 99999),
}
CAREER_BAND_RANGES = {}      # canon 名 → (lo, hi)  元ヘッダから解析した結果

# 通算成績のスナップショット時点（学習時に自動判定。"pre" / "post" / "leak" / "none"）
CAREER_SNAPSHOT = "post"
# 通算系特徴を除外するか（--drop-career または leak 判定時に True）
DROP_CAREER = False
# 除外対象の特徴名パターン
CAREER_FEATURE_PATTERN = re.compile(
    r"^通算|^脚質_|クラス変化|^レース内_(逃げ|先行|差し|追込)馬数|"
    r"^レース_(逃げ先行率|ハイペース指標)|^展開_|^騎手_同脚質_")


# ============================================================
# パース補助関数
# ============================================================
def decode_url_code(code):
    """URLコード yyppkkddrr を人間可読な情報に分解する。"""
    code = str(code).strip()
    if len(code) != 10 or not code.isdigit():
        return {"年": "", "競馬場番号": "", "競馬場名": "", "回": "",
                "日": "", "レース番号": "", "表示名": code}
    yy, pp, kk, dd, rr = (int(code[0:2]), int(code[2:4]), int(code[4:6]),
                          int(code[6:8]), int(code[8:10]))
    year = 2000 + yy
    place_name = PLACE_NAMES.get(pp, f"競馬場{pp:02d}")
    disp = f"{year}年 第{kk}回{place_name}{dd}日目 {rr}R"
    return {"年": year, "競馬場番号": pp, "競馬場名": place_name, "回": kk,
            "日": dd, "レース番号": rr, "表示名": disp}


def norm_str(value):
    if pd.isna(value):
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip()
    return s if s else None


def norm_track(value):
    s = norm_str(value)
    if s is None:
        return "NA"
    if "芝" in s:
        return "芝"
    if "ダ" in s:
        return "ダート"
    if "障" in s:
        return "障害"
    return s


def parse_style_numeric(value):
    """脚質文字列を順序尺度に変換（逃=1, 先=2, 差=3, 追=4）"""
    s = norm_str(value)
    if s is None:
        return np.nan
    return {"逃": 1.0, "先": 2.0, "差": 3.0, "追": 4.0, "マ": 3.0, "捲": 3.0}.get(s[0], np.nan)


def place_number(value):
    s = norm_str(value)
    if s is None:
        return np.nan
    if s in PLACE_ABBR:
        return float(PLACE_ABBR[s])
    for num, name in PLACE_NAMES.items():
        if s.startswith(name):
            return float(num)
    return parse_num(s)


def match_flag(a, b):
    a = pd.Series(a)
    b = pd.Series(b, index=a.index) if not isinstance(b, pd.Series) else b
    valid = a.notna() & b.notna()
    return (a == b).astype(float).where(valid)


def parse_finish(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else np.nan


def parse_weight_carried(value):
    if pd.isna(value):
        return np.nan, 0
    s = str(value).strip()
    allowance = 1 if re.match(r"^[▲☆△★◇]", s) else 0
    m = re.search(r"([\d.]+)", s)
    val = float(m.group(1)) if m else np.nan
    return val, allowance


def parse_num(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(m.group(0)) if m else np.nan


def parse_style_counts(value):
    """脚質:逃先差追 '00010102' -> [0,1,1,2] (逃,先,差,追)"""
    s = re.sub(r"\D", "", str(value)) if not pd.isna(value) else ""
    s = s.zfill(8)[:8]
    try:
        return [int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[6:8])]
    except ValueError:
        return [0, 0, 0, 0]


def parse_career(value):
    """'2-3-1-4' -> [win, place2, place3, total_extra]"""
    if pd.isna(value):
        return [0, 0, 0, 0]
    parts = re.findall(r"\d+", str(value))
    parts = (parts + [0, 0, 0, 0])[:4]
    return [int(p) for p in parts]


def parse_passing(value):
    if pd.isna(value):
        return [np.nan] * 4
    s = re.sub(r"\D", "", str(value))
    s = s.zfill(8)[:8]
    out = []
    for i in range(0, 8, 2):
        v = int(s[i:i + 2])
        out.append(np.nan if v == 0 else v)
    return out


def parse_distance_band(header):
    """距離帯列のヘッダ（例 '~1400', '1401-1800', '1801~', '1400以下'）から (lo, hi) を推定。
    解析不能なら None。"""
    s = unicodedata.normalize("NFKC", str(header)).strip()
    nums = [int(x) for x in re.findall(r"\d{3,4}", s)]
    if not nums:
        return None
    if len(nums) >= 2:
        return (min(nums), max(nums))
    n = nums[0]
    if re.search(r"以下|未満|まで", s) or re.match(r"^[~〜\-−–]", s):
        return (0, n)
    if re.search(r"以上|超|から", s) or re.search(r"[~〜\-−–]$", s):
        return (n, 99999)
    return None


def career_band_range(canon_col):
    """canon 列名 → 距離帯 (lo, hi)。通算_全 は None（全距離）。"""
    if canon_col == "通算_全":
        return None
    return CAREER_BAND_RANGES.get(canon_col) or CAREER_DIST_BANDS_DEFAULT.get(canon_col)


RACE_CLASSES = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "G3", "G2", "G1"]
RACE_CLASS_ORD = {c: i for i, c in enumerate(RACE_CLASSES)}


def parse_race_class(value):
    s = str(value)
    if re.search(r"G1|GI|GⅠ|ＧＩ", s): return "G1"
    if re.search(r"G2|GII|GⅡ|ＧⅡ", s): return "G2"
    if re.search(r"G3|GIII|GⅢ|ＧⅢ", s): return "G3"
    if re.search(r"オープン|OP|リステッド|L\b", s): return "OP"
    if re.search(r"1600万|3勝", s): return "3勝"
    if re.search(r"1000万|2勝", s): return "2勝"
    if re.search(r"500万|1勝", s): return "1勝"
    if re.search(r"未勝利", s): return "未勝利"
    if re.search(r"新馬", s): return "新馬"
    return "OP"


def infer_current_class_from_wins(wins_adj):
    """当該レース分を差し引いた通算1着数（レース前の勝ち数）の中央値からクラスを推定"""
    wins = pd.Series(np.asarray(wins_adj, dtype=float))
    med = wins.median()
    if pd.isna(med):
        return None
    if med <= 0:
        return float(RACE_CLASS_ORD["未勝利"])
    if med <= 1:
        return float(RACE_CLASS_ORD["1勝"])
    if med <= 2:
        return float(RACE_CLASS_ORD["2勝"])
    if med <= 3:
        return float(RACE_CLASS_ORD["3勝"])
    return float(RACE_CLASS_ORD["OP"])


def dominant_style_label(val):
    counts = parse_style_counts(val)
    if sum(counts) == 0:
        return "不明"
    styles = ["逃げ", "先行", "差し", "追込"]
    return styles[counts.index(max(counts))]


# ============================================================
# レース内相対特徴のヘルパー
# ============================================================
def group_top_and_second(v, grp_key):
    """レース内の最大値と2番目の値（同値トップが2頭以上なら2位=トップ）をベクトル化して返す"""
    g = v.groupby(grp_key)
    top = g.transform("max")
    n_top = (v == top).astype(float).where(v.notna(), 0.0).groupby(grp_key).transform("sum")
    second_cand = v.where(v < top).groupby(grp_key).transform("max")
    second = second_cand.where(n_top < 2, top)
    return top, second


# ============================================================
# 特徴量エンジニアリング
# ============================================================
USE_PACE_FEATURES = False

# 展開特徴の列名（一括削除用）
PACE_FEATURE_NAMES = [
    "レース内_差し馬数", "レース内_追込馬数",
    "レース_逃げ先行率", "レース_ハイペース指標",
    "展開_有利度", "展開_単騎逃げ", "展開_前の競馬数", "展開_先行率_レース内偏差",
]


def drop_pace_features(feat):
    return feat[[c for c in feat.columns if c not in PACE_FEATURE_NAMES]]


def build_features(df):
    feat = pd.DataFrame(index=df.index)

    # --- 基本数値 ---
    for c in BASE_NUM_FIELDS:
        feat[c] = df[c].apply(parse_num) if c in df.columns else np.nan

    place_num = df["競馬場"].apply(place_number) if "競馬場" in df.columns else pd.Series(np.nan, index=df.index)
    feat["競馬場名" + CAT_SUFFIX] = place_num.map(PLACE_NAMES_F)

    feat["レース番号"] = df["レース"].apply(parse_num) if "レース" in df.columns else np.nan

    # 性別のみカテゴリ化（年齢は数値特徴として別にあるため 性齢 は廃止）
    feat["性別" + CAT_SUFFIX] = df["性別"].map(norm_str) if "性別" in df.columns else None

    if "回り" in df.columns:
        feat["回り" + CAT_SUFFIX] = df["回り"].map(norm_str)

    # 血統情報（父馬・母父馬）をカテゴリ特徴として利用
    # ※ 母馬は産駒数が少なく過学習になりやすいため使用しない
    if "父馬" in df.columns:
        feat["父馬" + CAT_SUFFIX] = df["父馬"].map(norm_str)
    if "母父馬" in df.columns:
        feat["母父馬" + CAT_SUFFIX] = df["母父馬"].map(norm_str)

    feat["馬体重"] = df["馬体重"].apply(parse_num) if "馬体重" in df.columns else np.nan
    feat["体重増減"] = df["体重増減"].apply(parse_num) if "体重増減" in df.columns else np.nan

    if "斤量" in df.columns:
        kw = df["斤量"].apply(parse_weight_carried)
        feat["斤量"] = [x[0] for x in kw]
        feat["斤量減量フラグ"] = [x[1] for x in kw]
    else:
        feat["斤量"] = np.nan
        feat["斤量減量フラグ"] = 0

    baba_map = {"良": 1.0, "稍": 2.0, "稍重": 2.0, "重": 3.0, "不": 4.0, "不良": 4.0}
    if "馬場" in df.columns:
        feat["馬場_数値"] = df["馬場"].map(lambda x: baba_map.get(norm_str(x), np.nan))

    finish_now = df["着順"].apply(parse_finish) if "着順" in df.columns else None
    subtract_current = (finish_now is not None) and (CAREER_SNAPSHOT == "post")

    # --- 脚質傾向（脚質:逃先差追） ---
    if STYLE_COL in df.columns:
        sc = df[STYLE_COL].apply(parse_style_counts)
    else:
        sc = pd.Series([[0, 0, 0, 0]] * len(df), index=df.index)
    style_arr = np.array([list(x) for x in sc], dtype=float).reshape(len(df), 4)

    # レース後スナップショットなら当該レースの脚質分を差し引く
    if subtract_current and "脚質" in df.columns:
        style_idx = df["脚質"].apply(parse_style_numeric).values
        fn = finish_now.values
        for k in range(4):
            sel = (style_idx == k + 1) & ~np.isnan(fn)
            style_arr[sel, k] = np.maximum(style_arr[sel, k] - 1, 0)

    for k, name in enumerate(["逃", "先", "差", "追"]):
        feat[f"脚質_{name}"] = style_arr[:, k]
    total_style = pd.Series(style_arr.sum(axis=1), index=df.index).replace(0, np.nan)
    feat["脚質_先行率"] = (feat["脚質_逃"] + feat["脚質_先"]) / total_style
    feat["脚質_平均位置"] = (feat["脚質_逃"] * 1 + feat["脚質_先"] * 2 +
                        feat["脚質_差"] * 3 + feat["脚質_追"] * 4) / total_style

    # --- 通算成績 ---
    cur_dist = feat["距離"]
    career_wins_adj = None
    for i, col in enumerate(CAREER_CANON):
        if col in df.columns:
            parsed = df[col].apply(parse_career)
        else:
            parsed = pd.Series([[0, 0, 0, 0]] * len(df), index=df.index)
        w = np.array([x[0] for x in parsed], dtype=float)
        p2 = np.array([x[1] for x in parsed], dtype=float)
        p3 = np.array([x[2] for x in parsed], dtype=float)
        out = np.array([x[3] for x in parsed], dtype=float)

        if subtract_current:
            # 今回距離が該当する距離帯列（と 通算_全）のみ当該レース分を差し引く
            band = career_band_range(col)
            if band is None:
                applies = np.ones(len(df), dtype=bool)
            else:
                lo, hi = band
                applies = ((cur_dist >= lo) & (cur_dist <= hi)).fillna(False).values
            fn = finish_now.values
            valid = applies & ~np.isnan(fn)
            w = np.where(valid & (fn == 1), np.maximum(w - 1, 0), w)
            p2 = np.where(valid & (fn == 2), np.maximum(p2 - 1, 0), p2)
            p3 = np.where(valid & (fn == 3), np.maximum(p3 - 1, 0), p3)
            out = np.where(valid & (fn >= 4), np.maximum(out - 1, 0), out)

        if i == 0:
            career_wins_adj = w

        feat[f"通算{i}_1着"] = w
        feat[f"通算{i}_2着"] = p2
        feat[f"通算{i}_3着"] = p3
        feat[f"通算{i}_着外"] = out
        runs = w + p2 + p3 + out
        feat[f"通算{i}_出走"] = runs
        runs_safe = np.where(runs > 0, runs, np.nan)
        feat[f"通算{i}_勝率"] = w / runs_safe
        feat[f"通算{i}_複勝率"] = (w + p2 + p3) / runs_safe

    # --- 過去走ごとの特徴 ---
    cur_track = df["芝orダート"].apply(norm_track) if "芝orダート" in df.columns else pd.Series("NA", index=df.index)
    cur_place = place_num

    same_track_sp = []
    same_dist_sp = []

    for prefix in PAST_PREFIXES:
        for fld in PAST_NUMERIC_FIELDS:
            col = f"{prefix}の{fld}"
            feat[f"{prefix}_{fld}"] = df[col].apply(parse_num) if col in df.columns else np.nan

        if f"{prefix}の馬場" in df.columns:
            feat[f"{prefix}_馬場_数値"] = df[f"{prefix}の馬場"].map(lambda x: baba_map.get(norm_str(x), np.nan))
        if f"{prefix}の脚質" in df.columns:
            feat[f"{prefix}_脚質_数値"] = df[f"{prefix}の脚質"].apply(parse_style_numeric)

        interval_col = f"{prefix}からの日数"
        feat[f"{prefix}_間隔日数"] = df[interval_col].apply(parse_num) if interval_col in df.columns else np.nan

        feat[f"{prefix}_距離差"] = cur_dist - feat[f"{prefix}_距離"]

        pass_col = f"{prefix}の通過順位"
        if pass_col in df.columns:
            pp = df[pass_col].apply(parse_passing)
            feat[f"{prefix}_通過1"] = [x[0] for x in pp]
            feat[f"{prefix}_通過4"] = [x[3] for x in pp]
            heads = feat[f"{prefix}_頭数"]
            feat[f"{prefix}_通過4率"] = feat[f"{prefix}_通過4"] / heads.where(heads > 0)

        name_col = f"{prefix}のレース名"
        if name_col in df.columns:
            feat[f"{prefix}_クラス"] = df[name_col].apply(
                lambda v: RACE_CLASS_ORD.get(parse_race_class(v), np.nan))

        if f"{prefix}の芝orダート" in df.columns:
            past_track = df[f"{prefix}の芝orダート"].apply(norm_track)
            feat[f"{prefix}_同トラック"] = match_flag(past_track, cur_track)
        else:
            feat[f"{prefix}_同トラック"] = np.nan

        if f"{prefix}の競馬場" in df.columns:
            past_place = df[f"{prefix}の競馬場"].apply(place_number)
            feat[f"{prefix}_同競馬場"] = match_flag(past_place, cur_place)
        else:
            feat[f"{prefix}_同競馬場"] = np.nan

        sp = feat[f"{prefix}_スピード指数"]
        same_track_sp.append(sp.where(feat[f"{prefix}_同トラック"] == 1))
        same_dist_sp.append(sp.where(feat[f"{prefix}_距離差"].abs() <= 200))

    feat["同トラック_スピード指数平均"] = pd.concat(same_track_sp, axis=1).mean(axis=1)
    feat["同距離帯_スピード指数平均"] = pd.concat(same_dist_sp, axis=1).mean(axis=1)

    # --- 過去走の集約特徴 ---
    for fld in ["スピード指数", "先行指数", "ペース指数", "上がり指数", "着順", "人気",
                "タイム差", "上3F", "クラス"]:
        cols = [f"{p}_{fld}" for p in PAST_PREFIXES if f"{p}_{fld}" in feat.columns]
        if cols:
            sub = feat[cols]
            feat[f"{fld}_平均"] = sub.mean(axis=1)
            feat[f"{fld}_最大"] = sub.max(axis=1)
            feat[f"{fld}_最小"] = sub.min(axis=1)
            feat[f"{fld}_直近"] = feat[cols[0]]
            feat[f"{fld}_出走数"] = sub.notna().sum(axis=1)

    for fld in ["スピード指数", "着順", "上3F", "タイム差"]:
        cols = [f"{p}_{fld}" for p in PAST_PREFIXES[:3] if f"{p}_{fld}" in feat.columns]
        if cols:
            feat[f"{fld}_直近3走平均"] = feat[cols].mean(axis=1)

    fin_cols = [f"{p}_着順" for p in PAST_PREFIXES]
    feat["着順_標準偏差"] = feat[fin_cols].std(axis=1)

    interval_cols = [f"{p}_間隔日数" for p in PAST_PREFIXES]
    feat["前走からの日数"] = feat[interval_cols[0]]
    feat["間隔日数_平均"] = feat[interval_cols].mean(axis=1)

    feat["スピード指数_トレンド"] = feat["前走_スピード指数"] - feat["2走前_スピード指数"]
    feat["着順_トレンド"] = feat["前走_着順"] - feat["2走前_着順"]

    if "前走_クラス" in feat.columns and career_wins_adj is not None:
        if "URLコード" in df.columns:
            wins = pd.Series(career_wins_adj, index=df.index)
            cur_class = wins.groupby(df["URLコード"].astype(str)).transform(
                infer_current_class_from_wins)
        else:
            cur_class = infer_current_class_from_wins(career_wins_adj)
        if cur_class is not None:
            feat["クラス変化"] = cur_class - feat["前走_クラス"]

    d = feat["前走からの日数"]
    feat["中1週以内"] = (d <= 14).astype(float).where(d.notna())
    feat["休み明け60日超"] = (d > 60).astype(float).where(d.notna())

    # --- レース内相対特徴 ---
    if "URLコード" in df.columns:
        race_grp = df["URLコード"].astype(str)
        feat["レース頭数"] = race_grp.map(race_grp.value_counts()).astype(float)

        for col in ["スピード指数_直近", "スピード指数_平均", "スピード指数_最大",
                    "着順_直近3走平均", "前走_人気", "通算0_勝率", "通算0_複勝率",
                    "斤量", "年齢", "同トラック_スピード指数平均", "同距離帯_スピード指数平均",
                    "脚質_平均位置"]:
            if col in feat.columns:
                feat[f"{col}_レース内順位"] = feat.groupby(race_grp)[col].rank(
                    ascending=False, method="average")
                feat[f"{col}_レース内偏差"] = (
                    feat[col] - feat.groupby(race_grp)[col].transform("mean"))

        # 相手関係: トップとの差・2位との差（値が大きいほど良い向きに揃えてから計算）
        for col, sign in [("スピード指数_最大", 1), ("スピード指数_直近", 1),
                          ("スピード指数_平均", 1),
                          ("同距離帯_スピード指数平均", 1), ("同トラック_スピード指数平均", 1),
                          ("通算0_勝率", 1), ("通算0_複勝率", 1),
                          ("着順_直近3走平均", -1), ("前走_人気", -1)]:
            if col not in feat.columns:
                continue
            v = feat[col] * sign
            top, second = group_top_and_second(v, race_grp)
            feat[f"{col}_トップ差"] = v - top          # 0 = 自分がトップ、負 = トップとの差
            feat[f"{col}_2位差"] = v - second         # 自分がトップなら「2位を何点離しているか」

        # 展開: 他馬の逃げ馬数・先行馬数（自身を除く）
        has_style = style_arr.sum(axis=1) > 0
        dom = np.where(has_style, style_arr.argmax(axis=1), -1)
        is_nige = pd.Series((dom == 0).astype(float), index=df.index)
        is_senko = pd.Series((dom == 1).astype(float), index=df.index)
        is_sashi = pd.Series((dom == 2).astype(float), index=df.index)
        is_oikomi = pd.Series((dom == 3).astype(float), index=df.index)
        n_nige_others = is_nige.groupby(race_grp).transform("sum") - is_nige
        n_senko_others = is_senko.groupby(race_grp).transform("sum") - is_senko
        n_sashi_others = is_sashi.groupby(race_grp).transform("sum") - is_sashi
        n_oikomi_others = is_oikomi.groupby(race_grp).transform("sum") - is_oikomi
        feat["レース内_逃げ馬数"] = n_nige_others
        feat["レース内_先行馬数"] = n_senko_others
        feat["レース内_差し馬数"] = n_sashi_others
        feat["レース内_追込馬数"] = n_oikomi_others

        # --- 展開（ペース）予測の精緻化 ---
        # 逃げ・先行馬が多いほどハイペースになりやすい → 前残りが不利/差し追込が有利
        nige_total = n_nige_others + is_nige          # レース内の逃げ馬総数
        senko_total = n_senko_others + is_senko
        heads = feat["レース頭数"].replace(0, np.nan)
        feat["レース_逃げ先行率"] = (nige_total + senko_total) / heads
        feat["レース_ハイペース指標"] = (nige_total * 2 + senko_total) / heads
        # 自身の脚質とペースの相性:
        #   ハイペース × 差し/追込 → 有利 (+), ハイペース × 逃げ/先行 → 不利 (-)
        feat["展開_有利度"] = feat["レース_ハイペース指標"] * (
            is_sashi + is_oikomi - is_nige - is_senko)
        # 逃げ馬がいないレースでは逃げ馬が圧倒的に有利（単騎逃げ）
        feat["展開_単騎逃げ"] = ((n_nige_others == 0) & (is_nige == 1)).astype(float)
        # 自分より前に行きたがる馬の数（逃げ + 先行）= 位置取りの混戦度
        feat["展開_前の競馬数"] = n_nige_others + n_senko_others
        # 通算先行率のレース内平均との差（このレースで普段より前に行く必要があるか）
        if "脚質_先行率" in feat.columns:
            feat["展開_先行率_レース内偏差"] = (
                feat["脚質_先行率"] - feat.groupby(race_grp)["脚質_先行率"].transform("mean"))

    return feat


def drop_career_features(feat):
    """通算系（スクレイプ時点が不明・リーク疑いの列）を除外"""
    cols = [c for c in feat.columns if not CAREER_FEATURE_PATTERN.search(c)]
    return feat[cols]


# ============================================================
# 通算成績のスナップショット時点の診断
# ============================================================
def detect_career_snapshot(train_df):
    """通算_全 / 脚質:逃先差追 がいつの時点の値かを推定する。
        pre  : 出走数 0 の行が一定数ある（新馬戦の馬）→ レース前時点の値
        post : 同一馬の値がレースごとに異なる          → レース直後の値（差し引きが必要）
        leak : 同一馬の値が複数レースで同一             → 後日まとめて取得（未来の成績を含む）
    """
    if "通算_全" not in train_df.columns:
        print("  [通算診断] 通算_全 列がありません → 通算系特徴は全て 0 扱い")
        return "none"

    parsed = train_df["通算_全"].apply(parse_career)
    total = parsed.apply(sum).astype(float)
    zero_ratio = float((total == 0).mean())

    identical_ratio = np.nan
    n_multi = 0
    if "馬名" in train_df.columns:
        tmp = pd.DataFrame({"h": train_df["馬名"].astype(str), "t": total})
        g = tmp.groupby("h")["t"].agg(["size", "nunique"])
        multi = g[g["size"] >= 2]
        n_multi = len(multi)
        if n_multi:
            identical_ratio = float((multi["nunique"] == 1).mean())

    if STYLE_COL in train_df.columns:
        style_total = train_df[STYLE_COL].apply(lambda v: sum(parse_style_counts(v))).astype(float)
        diff = (style_total - total)
        print(f"  [通算診断] 脚質集計の合計 − 通算出走数: 中央値={diff.median():.1f} "
              f"(0 なら同一時点のスナップショット)")

    print(f"  [通算診断] 出走数0の行の割合={zero_ratio*100:.2f}%  "
          f"複数回出走馬 {n_multi} 頭のうち通算値が全レースで同一={identical_ratio*100 if not np.isnan(identical_ratio) else float('nan'):.1f}%")

    if zero_ratio >= 0.005:
        mode = "pre"
        print("  [通算診断] → レース前時点の値と判定。差し引きは行いません。")
    elif not np.isnan(identical_ratio) and identical_ratio > 0.3:
        mode = "leak"
        print("  [通算診断] → 【警告】同一馬の通算値が複数レースで一致。後日取得された値（未来の成績を含む）"
              "の可能性が高く、リークになります。通算系特徴を除外します。")
    else:
        mode = "post"
        print("  [通算診断] → レース直後の値と判定。当該レース分を差し引きます。")
    return mode


# ============================================================
# 騎手・調教師の実績エンコーディング（Target Encoding）
# ============================================================
def calc_personnel_stats(df):
    stats = {}
    if "_finish" not in df.columns:
        return stats

    temp = df[["騎手", "調教師", "_finish"]].copy() if {"騎手", "調教師"}.issubset(df.columns) \
        else pd.DataFrame({"_finish": df["_finish"]})
    temp["is_win"] = (temp["_finish"] == 1).astype(int)
    temp["is_top3"] = (temp["_finish"] <= 3).astype(int)

    temp["競馬場"] = df["競馬場"].astype(str) if "競馬場" in df.columns else "不明"
    temp["芝orダート"] = df["芝orダート"].apply(norm_track) if "芝orダート" in df.columns else "不明"
    temp["脚質"] = df[STYLE_COL].apply(dominant_style_label) if STYLE_COL in df.columns else "不明"

    for col in ["騎手", "調教師"]:
        if col not in temp.columns:
            continue
        grp = temp.groupby(col).agg(runs=("_finish", "count"), wins=("is_win", "sum"), top3=("is_top3", "sum"))
        col_stats = {"overall": {}}
        for idx, row in grp.iterrows():
            r = row["runs"]
            if r >= 3:
                col_stats["overall"][str(idx)] = {
                    "runs": int(r), "win_rate": float(row["wins"] / r), "top3_rate": float(row["top3"] / r)}
        stats[col] = col_stats

    if "騎手" in temp.columns:
        for cond_col in ["競馬場", "芝orダート", "脚質"]:
            cond_stats = {}
            grp = temp.groupby(["騎手", cond_col]).agg(
                runs=("_finish", "count"), wins=("is_win", "sum"), top3=("is_top3", "sum"))
            for (jockey, cond_val), row in grp.iterrows():
                r = row["runs"]
                if r >= 2:
                    cond_stats.setdefault(str(jockey), {})[str(cond_val)] = {
                        "runs": int(r), "win_rate": float(row["wins"] / r), "top3_rate": float(row["top3"] / r)}
            stats["騎手"][cond_col] = cond_stats

    return stats


def add_personnel_features(feat_df, orig_df, stats):
    feat = feat_df.copy()

    current_place = orig_df["競馬場"].astype(str) if "競馬場" in orig_df.columns else pd.Series("不明", index=orig_df.index)
    current_track = orig_df["芝orダート"].apply(norm_track) if "芝orダート" in orig_df.columns else pd.Series("不明", index=orig_df.index)
    current_style = orig_df[STYLE_COL].apply(dominant_style_label) if STYLE_COL in orig_df.columns \
        else pd.Series("不明", index=orig_df.index)

    for col in ["騎手", "調教師"]:
        if col not in orig_df.columns or col not in stats:
            feat[f"{col}_出走数"] = np.nan
            feat[f"{col}_勝率"] = np.nan
            feat[f"{col}_複勝率"] = np.nan
            continue

        overall = stats[col].get("overall", {})

        def get_stat(name, key):
            s = overall.get(str(name))
            return s[key] if s else np.nan

        names = orig_df[col].values
        feat[f"{col}_出走数"] = [get_stat(n, "runs") for n in names]
        feat[f"{col}_勝率"] = [get_stat(n, "win_rate") for n in names]
        feat[f"{col}_複勝率"] = [get_stat(n, "top3_rate") for n in names]

    if "騎手" in orig_df.columns and "騎手" in stats and "競馬場" in stats["騎手"]:
        names = orig_df["騎手"].values
        for cond_key, cond_vals, tag in [("競馬場", current_place.values, "同競馬場"),
                                         ("芝orダート", current_track.values, "同トラック"),
                                         ("脚質", current_style.values, "同脚質")]:
            cs = stats["騎手"].get(cond_key, {})
            feat[f"騎手_{tag}_出走数"] = [cs.get(str(n), {}).get(str(v), {}).get("runs", np.nan) for n, v in zip(names, cond_vals)]
            feat[f"騎手_{tag}_勝率"] = [cs.get(str(n), {}).get(str(v), {}).get("win_rate", np.nan) for n, v in zip(names, cond_vals)]
            feat[f"騎手_{tag}_複勝率"] = [cs.get(str(n), {}).get(str(v), {}).get("top3_rate", np.nan) for n, v in zip(names, cond_vals)]

    return feat


def add_personnel_features_oof(feat, train_df, groups, n_folds=5):
    """学習データ用 Out-of-Fold Target Encoding。
    時系列検証年（TIME_SPLIT_YEAR 以降）の行は、それより前の年の統計のみで符号化する。"""
    n = len(feat)
    years = pd.to_numeric(groups.astype(str).str.slice(0, 2), errors="coerce").fillna(-1).astype(int).values
    is_va = (years >= TIME_SPLIT_YEAR) if TIME_SPLIT_YEAR is not None else np.zeros(n, dtype=bool)
    tr_pos = np.where(~is_va)[0]
    va_pos = np.where(is_va)[0]
    if len(tr_pos) < n_folds:
        tr_pos, va_pos = np.arange(n), np.array([], dtype=int)

    parts = []
    sub_groups = groups.iloc[tr_pos].values
    k = min(n_folds, len(np.unique(sub_groups)))
    if k < 2:
        stats = calc_personnel_stats(train_df.iloc[tr_pos])
        return add_personnel_features(feat, train_df, stats)

    gkf = GroupKFold(n_splits=k)
    for k_tr, k_va in gkf.split(np.zeros(len(tr_pos)), groups=sub_groups):
        fit_pos, enc_pos = tr_pos[k_tr], tr_pos[k_va]
        stats = calc_personnel_stats(train_df.iloc[fit_pos])
        parts.append(add_personnel_features(feat.iloc[enc_pos], train_df.iloc[enc_pos], stats))
    if len(va_pos):
        stats = calc_personnel_stats(train_df.iloc[tr_pos])
        parts.append(add_personnel_features(feat.iloc[va_pos], train_df.iloc[va_pos], stats))
    return pd.concat(parts).loc[feat.index]


# ============================================================
# カテゴリ列のエンコード（XGBoost ネイティブカテゴリ）
# ============================================================
def fit_category_maps(feat):
    maps = {}
    for c in feat.columns:
        if c.endswith(CAT_SUFFIX):
            vc = feat[c].dropna().astype(str).value_counts()
            keep = vc[vc >= MIN_CAT_COUNT].index.tolist() or vc.index.tolist()
            maps[c] = sorted(keep) if keep else ["NA"]
    return maps


def apply_category_maps(feat, maps):
    feat = feat.copy()
    cat_cols = []
    for c in feat.columns:
        if c.endswith(CAT_SUFFIX):
            vals = feat[c].map(lambda v: None if pd.isna(v) else str(v))
            feat[c] = pd.Categorical(vals, categories=maps.get(c, ["NA"]))
            cat_cols.append(c)
    return feat, cat_cols


# ============================================================
# データ読み込み
# ============================================================
def normalize_career_columns(df):
    """脚質:逃先差追 の直後4列を共通名にリネームし、距離帯列の範囲を元ヘッダから解析する"""
    cols = list(df.columns)
    if STYLE_COL not in cols:
        return df
    idx = cols.index(STYLE_COL)
    career_src = cols[idx + 1: idx + 1 + 4]
    rename = {}
    for i, c in enumerate(career_src):
        if i < len(CAREER_CANON):
            rename[c] = CAREER_CANON[i]
            if i >= 1 and CAREER_CANON[i] not in CAREER_BAND_RANGES:
                band = parse_distance_band(c)
                if band is not None:
                    CAREER_BAND_RANGES[CAREER_CANON[i]] = band
    return df.rename(columns=rename)


def load_dir(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        except Exception as e:
            print(f"[読込失敗] {f}: {e}")
            continue
        if df.empty or "URLコード" not in df.columns:
            continue
        frames.append(normalize_career_columns(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# 学習（XGBoost LambdaRank / GPU）
# ============================================================
N_SPLITS = 3
MAX_ROUNDS = 15000
EARLY_STOP = 100
TIME_SPLIT_YEAR = 25   # 2025以降を検証用に時系列分割
RANDOM_SEED = 42

XGB_DEFAULT_PARAMS = dict(
    objective="rank:pairwise",
    eval_metric="ndcg@3",
    lambdarank_pair_method="topk",
    lambdarank_num_pair_per_sample=8,
    ndcg_exp_gain=0,               # ラベル（頭数−着順）を線形ゲインとして扱う
    tree_method="hist",
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=30,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.6,
    colsample_bynode=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0,
    max_bin=256,
    max_cat_to_onehot=16,
    random_state=RANDOM_SEED,
    verbosity=0,
)

# xgb_params() で必ず上書きする固定項目（tuned_params.json に旧設定が残っていても安全）
FIXED_PARAMS = {"objective", "eval_metric", "tree_method", "device", "random_state", "verbosity", "ndcg_exp_gain"}


def detect_device():
    try:
        Xt = np.random.rand(64, 4).astype(np.float32)
        yt = (np.random.rand(64) > 0.5).astype(int)
        dt = xgb.DMatrix(Xt, label=yt)
        xgb.train({"objective": "binary:logistic", "device": "cuda",
                   "tree_method": "hist", "verbosity": 0}, dt, num_boost_round=2)
        return "cuda"
    except Exception as e:
        print(f"  [GPU検査] CUDA利用不可のためCPUで学習します: {str(e)[:120]}")
        return "cpu"


DEVICE = None


def load_tuned_params():
    path = os.path.join(MODEL_DIR, TUNED_PARAMS_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def xgb_params(overrides=None):
    params = dict(XGB_DEFAULT_PARAMS)
    tuned = load_tuned_params()
    for k, v in tuned.items():
        if k not in FIXED_PARAMS:
            params[k] = v
    if overrides:
        params.update(overrides)
    params["objective"] = XGB_DEFAULT_PARAMS["objective"]
    params["eval_metric"] = XGB_DEFAULT_PARAMS["eval_metric"]
    params["ndcg_exp_gain"] = XGB_DEFAULT_PARAMS["ndcg_exp_gain"]
    params["tree_method"] = "hist"
    params["device"] = DEVICE
    params["random_state"] = RANDOM_SEED
    params["verbosity"] = 0
    return params


def make_folds(groups):
    if TIME_SPLIT_YEAR is not None:
        years = pd.to_numeric(groups.str.slice(0, 2), errors="coerce").fillna(-1).astype(int)
        tr = np.where(years < TIME_SPLIT_YEAR)[0]
        va = np.where(years >= TIME_SPLIT_YEAR)[0]
        if len(tr) > 0 and len(va) > 0:
            return [(tr, va)]
        print("  [警告] 時系列分割に失敗したため GroupKFold にフォールバックします")
    gkf = GroupKFold(n_splits=N_SPLITS)
    return list(gkf.split(np.zeros(len(groups)), np.zeros(len(groups)), groups))


# ------------------------------------------------------------
# ランキング評価・確率化（Plackett–Luce）
# ------------------------------------------------------------
def evaluate_ranking(scores, finish, groups):
    """予測スコア最上位馬の 1着的中率・3着以内率（勝者が含まれるレースのみ）"""
    df = pd.DataFrame({"g": np.asarray(groups), "f": np.asarray(finish, dtype=float),
                       "s": np.asarray(scores, dtype=float)})
    df = df[df["s"].notna()]
    has_winner = df.groupby("g")["f"].transform("min") == 1
    df = df[has_winner]
    if df.empty:
        return float("nan"), float("nan"), 0
    top = df.loc[df.groupby("g")["s"].idxmax()]
    return float((top["f"] == 1).mean()), float((top["f"] <= 3).mean()), len(top)


def blend_group_scores(scores, groups, optional_scores=None, weight=0.0,
                       win_probabilities=False):
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    if scores.ndim != 1 or groups.shape != scores.shape or pd.isna(groups).any():
        raise ValueError("Scores and race groups must be matching one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Scores and blend weight must be finite; weight must be in [0, 1]")
    if weight == 0:
        return scores.copy()
    if optional_scores is None:
        raise ValueError("An optional model is required for a nonzero blend weight")
    optional = np.asarray(optional_scores, dtype=float)
    if optional.shape != scores.shape or not np.isfinite(optional).all():
        raise ValueError("Optional scores must be finite and match ranking scores")
    if win_probabilities:
        if ((optional < 0) | (optional > 1)).any():
            raise ValueError("Win classifier probabilities must be in [0, 1]")
        optional = np.clip(optional, 1e-9, 1 - 1e-9)
        optional = np.log(optional / (1 - optional))
    mixed = np.empty_like(scores)
    for positions in pd.Series(np.arange(len(scores))).groupby(groups, sort=False).indices.values():
        base = scores[positions]
        other = optional[positions]
        base = (base - base.mean()) / (base.std(ddof=0) + 1e-9)
        other = (other - other.mean()) / (other.std(ddof=0) + 1e-9)
        mixed[positions] = (1 - weight) * base + weight * other
    return mixed


def select_blend(scores, finish, groups, candidates):
    selected = blend_group_scores(scores, groups)
    best_hit, _, _ = evaluate_ranking(selected, finish, groups)
    best_name, best_weight = None, 0.0
    for name, optional, weights in candidates:
        for weight in weights:
            mixed = blend_group_scores(scores, groups, optional, float(weight),
                                       win_probabilities=name == "win")
            hit, _, _ = evaluate_ranking(mixed, finish, groups)
            if hit > best_hit:
                best_name, best_weight, best_hit = name, float(weight), hit
                selected = mixed
    temperature, _ = fit_temperature(selected, finish, groups)
    return best_name, best_weight, selected, temperature, best_hit


def softmax_by_group(scores, groups, temperature):
    s = pd.Series(np.asarray(scores, dtype=float) / temperature)
    g = pd.Series(np.asarray(groups))
    z = s - s.groupby(g).transform("max")
    w = np.exp(z)
    return (w / w.groupby(g).transform("sum")).values


def fit_temperature(scores, finish, groups):
    """レース内 softmax の温度を 1着 LogLoss 最小で選ぶ"""
    y = (np.asarray(finish, dtype=float) == 1)
    best_T, best_ll = 1.0, np.inf
    for T in np.logspace(-1.5, 1.5, 61):
        p = softmax_by_group(scores, groups, T)
        ll = -np.mean(np.log(np.clip(p[y], 1e-9, 1.0)))
        if ll < best_ll:
            best_ll, best_T = ll, float(T)
    return best_T, float(best_ll)


def pl_topk_probs(scores, temperature):
    """1レースのスコア → Plackett–Luce による 1着 / 2着以内 / 3着以内 確率（厳密解）"""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("Scores must be a finite one-dimensional array")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    n = len(s)
    if n == 0:
        return s.copy(), s.copy(), s.copy()

    def probabilities(indices):
        values = s[indices]
        weights = np.exp((values - values.max()) / temperature)
        return weights / weights.sum()

    indices = np.arange(n)
    p1 = probabilities(indices)
    if n <= 2:
        return p1, np.ones(n), np.ones(n)
    p2 = p1.copy()
    third = np.zeros(n)
    for first in indices:
        remaining = indices[indices != first]
        second_probs = probabilities(remaining)
        p2[remaining] += p1[first] * second_probs
        if n > 3 and p1[first] > 0:
            for second, conditional in zip(remaining, second_probs):
                last = remaining[remaining != second]
                third[last] += p1[first] * conditional * probabilities(last)
    p2 = np.clip(p2, 0, 1)
    p3 = np.clip(p2 + third, 0, 1) if n > 3 else np.ones(n)
    return p1, p2, p3


def print_baselines(X, finish, groups, mask):
    """単純ベースラインとの比較（モデルがこれを大幅に上回りすぎる場合はリークを疑う）"""
    Xm, fm, gm = X[mask], finish[mask], groups[mask]
    for name, col, sign in [("前走人気が最も高い馬", "前走_人気", -1),
                            ("直近スピード指数が最大の馬", "スピード指数_直近", 1),
                            ("スピード指数最大値が最大の馬", "スピード指数_最大", 1)]:
        if col not in Xm.columns:
            continue
        s = (Xm[col].astype(float) * sign).fillna(-1e9).values
        h1, h3, n = evaluate_ranking(s, fm.values, gm.values)
        print(f"    ベースライン[{name}] 1着的中率={h1:.4f}  3着内率={h3:.4f}  (n_races={n})")


def report_categorical_importance(model, cat_cols):
    scores = model.get_score(importance_type="gain")
    total = sum(scores.values()) or 1.0
    cat_gain = sum(v for k, v in scores.items() if k in cat_cols)
    used = [k for k in cat_cols if scores.get(k, 0) > 0]
    print(f"    カテゴリ特徴の重要度シェア: {cat_gain / total * 100:.1f}%  "
          f"(採用 {len(used)}/{len(cat_cols)} 列)")


def make_rank_label(finish, groups):
    """ランキング用ラベル: 頭数 − 着順（大きいほど良い、下限 0）"""
    n_heads = groups.map(groups.value_counts()).astype(float)
    return (n_heads - finish.astype(float)).clip(lower=0)


def train_lgb_rank(X, label, groups, cat_cols, tr, va, seed=RANDOM_SEED):
    """LightGBM の lambdarank で学習し (model, best_iter, valスコア) を返す"""
    import lightgbm as lgb
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3],
        lambdarank_truncation_level=18,
        label_gain=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_data_in_leaf=40,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        verbose=-1,
        seed=seed,
    )
    dtr = lgb.Dataset(X.iloc[tr], label=label.iloc[tr],
                      group=groups.iloc[tr].value_counts().sort_index().values,
                      categorical_feature=cat_cols if cat_cols else "auto")
    dva = lgb.Dataset(X.iloc[va], label=label.iloc[va],
                      group=groups.iloc[va].value_counts().sort_index().values,
                      reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    pred = model.predict(X.iloc[va], num_iteration=model.best_iteration)
    return model, model.best_iteration, pred


def group_sizes(groups):
    """グループごとの行数（LightGBM の group パラメータ用）"""
    return groups.value_counts().sort_index().values


def train_and_eval(X, finish, groups, cat_cols, param_overrides=None,
                   use_lgb=False, use_mt=False):
    X = X.reset_index(drop=True)
    finish = finish.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    label = make_rank_label(finish, groups)
    qid = pd.factorize(groups)[0]          # groups は URLコード順にソート済み → 連続

    folds = make_folds(groups)
    oof = np.full(len(X), np.nan)
    best_iters = []

    for fold, (tr, va) in enumerate(folds):
        dtr = xgb.DMatrix(X.iloc[tr], label=label.iloc[tr], qid=qid[tr], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va], label=label.iloc[va], qid=qid[va], enable_categorical=True)
        model = xgb.train(
            xgb_params(param_overrides), dtr, num_boost_round=MAX_ROUNDS,
            evals=[(dva, "valid")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        oof[va] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        best_iters.append(model.best_iteration + 1)
        print(f"    fold{fold+1}/{len(folds)} best_iter={model.best_iteration + 1}", flush=True)

    mask = ~np.isnan(oof)
    hit1, hit3, nrace = evaluate_ranking(oof[mask], finish[mask].values, groups[mask].values)
    temperature, win_ll = fit_temperature(oof[mask], finish[mask].values, groups[mask].values)
    print(f"  [ランキング] 予測1位の1着的中率={hit1:.4f}  予測1位の3着内率={hit3:.4f}  "
          f"(n_races={nrace})")
    print(f"  [較正] softmax温度={temperature:.3f}  1着LogLoss={win_ll:.4f}")
    print_baselines(X, finish, groups, mask)

    # 検証データでの確信度別的中率（レースを絞った場合の目安）
    p_win = softmax_by_group(oof[mask], groups[mask].values, temperature)
    sel = pd.DataFrame({"g": groups[mask].values, "f": finish[mask].values, "p": p_win})
    top = sel.loc[sel.groupby("g")["p"].idxmax()]
    for th in [0.0, 0.3, 0.4, 0.5, 0.6]:
        sub = top[top["p"] >= th]
        if len(sub):
            print(f"    1位確率>={th:.1f}: 対象{len(sub)/len(top)*100:5.1f}%のレース  "
                  f"1着的中率={(sub['f']==1).mean():.3f}  3着内率={(sub['f']<=3).mean():.3f}")

    n_final = int(np.mean(best_iters) * 1.1) if best_iters else 500
    dall = xgb.DMatrix(X, label=label, qid=qid, enable_categorical=True)
    final_model = xgb.train(xgb_params(param_overrides), dall,
                            num_boost_round=max(n_final, 50), verbose_eval=False)

    try:
        imp = final_model.get_score(importance_type="gain")
        top_imp = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:15]
        print("    重要度Top15: " + ", ".join(k for k, _ in top_imp))
    except Exception:
        pass
    report_categorical_importance(final_model, cat_cols)

    candidates = []
    lgb_iters, mt_iters = [], []
    lgb_model, mt_model = None, None
    ens_w, mt_w = 0.0, 0.0
    if use_lgb:
        lgb_oof = np.full(len(X), np.nan)
        for tr, va in folds:
            _, best_iter, pred = train_lgb_rank(X, label, groups, cat_cols, tr, va)
            lgb_oof[va] = pred
            lgb_iters.append(best_iter)
        candidates.append(("lgb", lgb_oof[mask], np.linspace(0.1, 0.9, 9)))
    if use_mt:
        mt_params = dict(objective="binary:logistic", eval_metric="logloss",
                         tree_method="hist", device=DEVICE,
                         learning_rate=0.05, max_depth=5, min_child_weight=30,
                         subsample=0.8, colsample_bytree=0.6,
                         reg_alpha=1.0, reg_lambda=5.0,
                         random_state=RANDOM_SEED, verbosity=0)
        mt_oof = np.full(len(X), np.nan)
        for tr, va in folds:
            mtr = xgb.DMatrix(X.iloc[tr], label=(finish.iloc[tr] == 1).astype(int),
                              enable_categorical=True)
            mva = xgb.DMatrix(X.iloc[va], label=(finish.iloc[va] == 1).astype(int),
                              enable_categorical=True)
            m1 = xgb.train(mt_params, mtr, num_boost_round=3000,
                           evals=[(mva, "valid")], early_stopping_rounds=100,
                           verbose_eval=False)
            mt_oof[va] = m1.predict(mva, iteration_range=(0, m1.best_iteration + 1))
            mt_iters.append(m1.best_iteration + 1)
        candidates.append(("win", mt_oof[mask], np.linspace(0.05, 0.5, 10)))

    selected, weight, mixed, temperature, hit1 = select_blend(
        oof[mask], finish[mask].values, groups[mask].values, candidates)
    p_win = softmax_by_group(mixed, groups[mask].values, temperature)
    win_ll = -np.mean(np.log(np.clip(p_win[finish[mask].values == 1], 1e-9, 1)))
    print(f"  [最終選択] {selected or 'XGBoost'} 重み={weight:.2f} "
          f"1着的中率={hit1:.4f} 温度={temperature:.3f} LogLoss={win_ll:.4f}")
    if selected == "lgb":
        import lightgbm as lgb
        ens_w = weight
        dall_l = lgb.Dataset(X, label=label, group=group_sizes(groups),
                             categorical_feature=cat_cols if cat_cols else "auto")
        lgb_model = lgb.train(
            dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
                 lambdarank_truncation_level=18, label_gain=list(range(18)),
                 learning_rate=0.02, num_leaves=31, max_depth=6,
                 min_data_in_leaf=40, feature_fraction=0.7,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                 verbose=-1, seed=RANDOM_SEED),
            dall_l, num_boost_round=max(int(np.mean(lgb_iters) * 1.1), 50))
    elif selected == "win":
        mt_w = weight
        dall_m = xgb.DMatrix(X, label=(finish == 1).astype(int), enable_categorical=True)
        mt_model = xgb.train(mt_params, dall_m,
                             num_boost_round=max(int(np.mean(mt_iters) * 1.1), 50),
                             verbose_eval=False)

    return final_model, temperature, hit1, lgb_model, ens_w, mt_model, mt_w


# ============================================================
# 学習データ準備
# ============================================================
def load_training_frame():
    global CAREER_SNAPSHOT, DROP_CAREER
    print("過去データを読み込み中 ...")
    past = load_dir(PAST_DIR)
    if past.empty:
        raise SystemExit("CSV_past にデータがありません。")
    print(f"  レース数(URLコード): {past['URLコード'].nunique()},  行数: {len(past)}")
    past["_finish"] = past["着順"].apply(parse_finish)
    past = past[past["芝orダート"].map(norm_track) != "障害"].copy()
    train_df = past[past["_finish"].notna()].copy()
    train_df["_finish"] = train_df["_finish"].astype(int)
    # ランキング学習のため URLコード順にソート（qid が連続になる）
    train_df["URLコード"] = train_df["URLコード"].astype(str)
    train_df = train_df.sort_values("URLコード", kind="stable").reset_index(drop=True)

    heads = train_df.groupby("URLコード").size()
    print(f"  レースあたり行数: 平均={heads.mean():.1f}  最小={heads.min()}  最大={heads.max()}")
    if CAREER_BAND_RANGES:
        print(f"  距離帯列の範囲（ヘッダ解析）: {CAREER_BAND_RANGES}")
    else:
        print(f"  距離帯列の範囲（デフォルト）: {CAREER_DIST_BANDS_DEFAULT}")

    CAREER_SNAPSHOT = detect_career_snapshot(train_df)
    if CAREER_SNAPSHOT == "leak" and not DROP_CAREER:
        DROP_CAREER = True
    if DROP_CAREER:
        print("  [通算系特徴] 除外して学習します。")
    return train_df


def prepare_training_matrix(train_df):
    groups = train_df["URLコード"].astype(str)
    print("特徴量を生成中 ...")
    X_full = build_features(train_df)
    if DROP_CAREER:
        X_full = drop_career_features(X_full)
    if not USE_PACE_FEATURES:
        X_full = drop_pace_features(X_full)
    print("騎手・調教師の Target Encoding（OOF）を計算中 ...")
    X_full = add_personnel_features_oof(X_full, train_df, groups)
    if DROP_CAREER:
        X_full = drop_career_features(X_full)
    personnel_stats = calc_personnel_stats(train_df)
    cat_maps = fit_category_maps(X_full)
    X_enc, cat_cols = apply_category_maps(X_full, cat_maps)
    print(f"  特徴量数: {len(X_enc.columns)}  (うちカテゴリ {len(cat_cols)}: {cat_cols})")
    return X_enc, cat_cols, cat_maps, personnel_stats, groups


# ============================================================
# パラメータ自動探索（Optuna / --tune）
# ============================================================
TUNE_SEARCH_SPACE = dict(
    learning_rate=("float", 0.01, 0.08, "log"),
    max_depth=("int", 3, 7, None),
    min_child_weight=("float", 5.0, 100.0, "log"),
    gamma=("float", 0.0, 1.0, None),
    subsample=("float", 0.6, 0.95, None),
    colsample_bytree=("float", 0.4, 0.9, None),
    colsample_bynode=("float", 0.6, 1.0, None),
    reg_alpha=("float", 0.01, 10.0, "log"),
    reg_lambda=("float", 0.5, 20.0, "log"),
    max_cat_to_onehot=("int", 4, 32, None),
    lambdarank_num_pair_per_sample=("int", 2, 12, None),
)


def run_tuning(n_trials=50):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise SystemExit("Optuna が必要です:  python -m pip install optuna")

    print("チューニング用にデータを準備中 ...")
    train_df = load_training_frame()
    finish = train_df["_finish"].reset_index(drop=True)
    X_enc, cat_cols, cat_maps, personnel_stats, groups = prepare_training_matrix(train_df)
    X_enc = X_enc.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    label = make_rank_label(finish, groups)
    qid = pd.factorize(groups)[0]

    folds = make_folds(groups)
    tr_idx, va_idx = folds[0]
    dtr = xgb.DMatrix(X_enc.iloc[tr_idx], label=label.iloc[tr_idx], qid=qid[tr_idx], enable_categorical=True)
    dva = xgb.DMatrix(X_enc.iloc[va_idx], label=label.iloc[va_idx], qid=qid[va_idx], enable_categorical=True)
    f_va = finish.iloc[va_idx].values
    g_va = groups.iloc[va_idx].values

    def suggest(trial, space):
        params = {}
        for name, (kind, lo, hi, log) in space.items():
            if kind == "int":
                params[name] = trial.suggest_int(name, lo, hi)
            elif log == "log":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
            else:
                params[name] = trial.suggest_float(name, lo, hi)
        return params

    def objective(trial):
        overrides = suggest(trial, TUNE_SEARCH_SPACE)
        model = xgb.train(
            xgb_params(overrides), dtr, num_boost_round=MAX_ROUNDS,
            evals=[(dva, "valid")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        pred = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        hit1, _, _ = evaluate_ranking(pred, f_va, g_va)
        return hit1

    print(f"\nOptuna による探索を開始（{n_trials} 試行）...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    enqueue = {k: XGB_DEFAULT_PARAMS[k] for k in TUNE_SEARCH_SPACE if k in XGB_DEFAULT_PARAMS}
    study.enqueue_trial(enqueue)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print("\n" + "-" * 60)
    print(f"探索完了: ベスト的中率 = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    default_score = study.trials[0].value
    if best.value <= (default_score or 0):
        print(f"\n※ デフォルト値（的中率 {default_score:.4f}）を上回らなかったため、tuned_params.json は更新しません。")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, TUNED_PARAMS_FILE)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(best.params, fh, ensure_ascii=False, indent=2)
    print(f"\n最適パラメータを {out_path} に保存しました。")


# ============================================================
# モデルの保存・読込
# ============================================================
def validate_blend_models(lgb_model, ensemble_weight, mt_model, mt_weight):
    for model, weight in ((lgb_model, ensemble_weight), (mt_model, mt_weight)):
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Blend weights must be finite and in [0, 1]")
        if bool(model is not None) != bool(weight > 0):
            raise ValueError("Optional model and nonzero blend weight must both be present")
    if ensemble_weight > 0 and mt_weight > 0:
        raise ValueError("Only one optional blend model may be selected; retrain")


def save_models(model, cat_maps, feature_cols, cat_cols, personnel_stats, temperature,
                lgb_model=None, ensemble_weight=0.0, mt_model=None, mt_weight=0.0):
    validate_blend_models(lgb_model, ensemble_weight, mt_model, mt_weight)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, MODEL_FILE))
    if lgb_model is not None:
        lgb_model.save_model(os.path.join(MODEL_DIR, LGB_MODEL_FILE))
    if mt_model is not None:
        mt_model.save_model(os.path.join(MODEL_DIR, MT_MODEL_FILE))
    meta = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_maps": cat_maps,
        "personnel_stats": personnel_stats,
        "temperature": temperature,
        "career_snapshot": CAREER_SNAPSHOT,
        "drop_career": DROP_CAREER,
        "use_pace_features": USE_PACE_FEATURES,
        "career_band_ranges": {k: list(v) for k, v in CAREER_BAND_RANGES.items()},
        "params": xgb_params(),
        "has_lgb": lgb_model is not None,
        "ensemble_weight": ensemble_weight,
        "has_mt": mt_model is not None,
        "mt_weight": mt_weight,
        "version": META_VERSION,
    }
    with open(os.path.join(MODEL_DIR, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    print(f"  学習済みモデルを {MODEL_DIR} に保存しました。")


def load_models():
    global CAREER_SNAPSHOT, DROP_CAREER, USE_PACE_FEATURES
    meta_path = os.path.join(MODEL_DIR, META_FILE)
    if not os.path.exists(meta_path):
        raise SystemExit(f"学習済みモデルが見つかりません（{meta_path}）。\n"
                         f"先に学習を実行してください:  python predict_model.py --train")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("version", 1) != META_VERSION:
        raise SystemExit("保存済みモデルのスコア変換・温度較正は未対応です。再学習してください:\n"
                         "  python predict_model.py --train")
    try:
        validate_blend_models(True if meta.get("has_lgb") else None,
                              float(meta.get("ensemble_weight", 0)),
                              True if meta.get("has_mt") else None,
                              float(meta.get("mt_weight", 0)))
        temperature = float(meta["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid temperature")
    except (ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"保存済みモデルの較正情報が不正です。再学習してください: {exc}")
    for flag, filename in (("has_lgb", LGB_MODEL_FILE), ("has_mt", MT_MODEL_FILE)):
        optional_path = os.path.join(MODEL_DIR, filename)
        if meta.get(flag) and not os.path.isfile(optional_path):
            raise SystemExit(f"モデルファイルが見つかりません: {optional_path}。再学習してください。")

    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"モデルファイルが見つかりません: {path}")
    booster = xgb.Booster()
    booster.load_model(path)
    booster.set_param({"device": DEVICE})

    CAREER_SNAPSHOT = meta.get("career_snapshot", "post")
    DROP_CAREER = bool(meta.get("drop_career", False))
    USE_PACE_FEATURES = bool(meta.get("use_pace_features", False))
    for k, v in meta.get("career_band_ranges", {}).items():
        CAREER_BAND_RANGES.setdefault(k, tuple(v))

    lgb_model = None
    if meta.get("has_lgb"):
        lgb_path = os.path.join(MODEL_DIR, LGB_MODEL_FILE)
        import lightgbm as lgb
        lgb_model = lgb.Booster(model_file=lgb_path)

    mt_model = None
    if meta.get("has_mt"):
        mt_path = os.path.join(MODEL_DIR, MT_MODEL_FILE)
        mt_model = xgb.Booster()
        mt_model.load_model(mt_path)
        mt_model.set_param({"device": DEVICE})

    return (booster, meta["cat_maps"], meta["feature_cols"], meta["cat_cols"],
            meta.get("personnel_stats", {}), float(meta.get("temperature", 1.0)),
            lgb_model, float(meta.get("ensemble_weight", 0.0)),
            mt_model, float(meta.get("mt_weight", 0.0)))


# ============================================================
# 学習フェーズ
# ============================================================
def run_training(no_lgb=True, no_mt=True):
    train_df = load_training_frame()
    finish = train_df["_finish"]
    X_enc, cat_cols, cat_maps, personnel_stats, groups = prepare_training_matrix(train_df)
    feature_cols = list(X_enc.columns)

    tuned = load_tuned_params()
    if tuned:
        print(f"  [パラメータ] tuned_params.json を適用します: {tuned}")
    else:
        print(f"  [パラメータ] デフォルト値を使用します (depth={XGB_DEFAULT_PARAMS['max_depth']}, "
              f"lr={XGB_DEFAULT_PARAMS['learning_rate']}, mcw={XGB_DEFAULT_PARAMS['min_child_weight']})")

    print("\nランキングモデル学習 & 時系列検証 ...")
    model, temperature, _, lgb_model, ens_w, mt_model, mt_w = train_and_eval(
        X_enc, finish, groups, cat_cols, use_lgb=not no_lgb, use_mt=not no_mt)

    print("\n学習済みモデルを保存中 ...")
    save_models(model, cat_maps, feature_cols, cat_cols, personnel_stats, temperature,
                lgb_model=lgb_model, ensemble_weight=ens_w,
                mt_model=mt_model, mt_weight=mt_w)
    return (model, cat_maps, feature_cols, cat_cols, personnel_stats, temperature,
            lgb_model, ens_w, mt_model, mt_w)


# ============================================================
# 予測フェーズ
# ============================================================
def run_prediction(model, cat_maps, feature_cols, cat_cols, personnel_stats=None,
                   temperature=1.0, lgb_model=None, ensemble_weight=0.0,
                   mt_model=None, mt_weight=0.0, race_code=None, output_dir=None):
    validate_blend_models(lgb_model, ensemble_weight, mt_model, mt_weight)
    personnel_stats = personnel_stats or {}
    output_dir = os.fspath(output_dir) if output_dir is not None else OUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    if race_code is not None and not re.fullmatch(r"[0-9]{10}", race_code):
        raise ValueError("race-code must be a ten-digit URL code")

    print("\n" + "=" * 60)
    print("予測対象レースを処理中 ...")
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.csv")))
    if race_code is None and len(pred_files) > 1:
        print(f"  [警告] {len(pred_files)} 個の予測対象ファイルがあります。"
              f"単一レースに絞るには --race-code を使用してください。")
    if not pred_files:
        raise SystemExit("CSV_predict にデータがありません。")

    all_results = []
    seen_codes = set()

    for f in pred_files:
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        if df.empty:
            continue
        if "URLコード" not in df.columns:
            raise SystemExit(f"URLコード列がありません: {f}")
        codes = df["URLコード"].astype("string").str.strip()
        if codes.isna().any() or not codes.str.fullmatch(r"[0-9]{10}").all():
            raise SystemExit(f"URLコードが不正です: {f}")
        if race_code is not None:
            df = df.loc[codes == race_code].copy()
            if df.empty:
                continue
            codes = codes.loc[df.index]
        if codes.nunique() != 1:
            raise SystemExit(f"複数のURLコードがあります ({sorted(codes.unique())})。"
                             f"--race-code で指定してください: {f}")
        code = str(codes.iloc[0])
        if code in seen_codes:
            raise SystemExit(f"URLコードが複数ファイルにあります: {code}")
        seen_codes.add(code)
        df = normalize_career_columns(df).reset_index(drop=True)
        df["URLコード"] = code
        Xp = build_features(df)
        Xp = add_personnel_features(Xp, df, personnel_stats)

        for c in feature_cols:
            if c not in Xp.columns:
                Xp[c] = np.nan
        Xp = Xp[feature_cols]
        Xp_enc, _ = apply_category_maps(Xp, cat_maps)
        Xp_enc = Xp_enc[feature_cols]

        dmat = xgb.DMatrix(Xp_enc, enable_categorical=True)
        score = np.asarray(model.predict(dmat), dtype=float)
        optional = None
        if lgb_model is not None:
            optional = lgb_model.predict(Xp_enc)
        elif mt_model is not None:
            optional = mt_model.predict(dmat)
        score = blend_group_scores(score, df["URLコード"].values, optional,
                                   ensemble_weight or mt_weight,
                                   win_probabilities=mt_model is not None)
        p_win, p_top2, p_top3 = pl_topk_probs(score, temperature)

        info = decode_url_code(code)

        res = pd.DataFrame({
            "開催日": f"{info['年']}年 第{info['回']}回{info['日']}日目",
            "競馬場": info["競馬場名"],
            "レース番号": f"{info['レース番号']}R",
            "URLコード": code,
            "枠番": df.get("枠番", pd.Series(np.nan, index=df.index)).apply(parse_num).astype("Int64"),
            "馬番": df["馬番"].apply(parse_num).astype("Int64"),
            "馬名": df["馬名"].astype(str),
            "騎手": df.get("騎手", ""),            "スコア": score,
            "1着確率": p_win,
            "2着以内確率": p_top2,
            "3着以内確率": p_top3,
        })
        all_results.append(res)

        track = df["芝orダート"].iloc[0] if "芝orダート" in df.columns else ""
        dist = df["距離"].iloc[0] if "距離" in df.columns else ""
        print(f"  処理完了: {info['表示名']} ({track}{dist}m {len(df)}頭)")

        out = res.copy()
        out["1着_順位"] = out["1着確率"].rank(ascending=False, method="min").astype(int)
        out["2着以内_順位"] = out["2着以内確率"].rank(ascending=False, method="min").astype(int)
        out["3着以内_順位"] = out["3着以内確率"].rank(ascending=False, method="min").astype(int)
        out = out.sort_values("1着確率", ascending=False)
        fname = f"pred_{info['年']}_第{info['回']}回{info['競馬場名']}{info['日']}日目_{info['レース番号']}R.csv"
        out.to_csv(os.path.join(output_dir, fname), index=False, encoding="utf-8-sig")

    if race_code is not None and not all_results:
        raise SystemExit(f"指定されたURLコードが見つかりません: {race_code}")
    if all_results:
        alldf = pd.concat(all_results, ignore_index=True)
        alldf.to_csv(os.path.join(output_dir, "_all_predictions.csv"), index=False, encoding="utf-8-sig")
        print("\n" + "=" * 60)
        print(f"完了。予測結果を {output_dir} に保存しました。")


# ============================================================
# メイン
# ============================================================
def main():
    global DEVICE, DROP_CAREER, USE_PACE_FEATURES

    parser = argparse.ArgumentParser(
        description="競馬 着順予測モデル（XGBoost LambdaRank / GPU）",
        formatter_class=argparse.RawTextHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="CSV_past で学習し直してモデルを保存する")
    mode.add_argument("--predict-only", action="store_true", help="保存済みモデルで CSV_predict を予測する")
    mode.add_argument("--tune", action="store_true", help="Optuna でハイパーパラメータを自動探索")
    parser.add_argument("--trials", type=int, default=50, help="--tune 時の探索試行回数（デフォルト: 50）")
    parser.add_argument("--drop-career", action="store_true",
                        help="通算成績・脚質集計（脚質:逃先差追）由来の特徴を使わない")
    parser.add_argument("--lgb", action="store_true",
                        help="実験的なLightGBMブレンド候補を検証する（デフォルトOFF）")
    parser.add_argument("--win-blend", action="store_true",
                        help="実験的な1着分類器ブレンド候補を検証する（デフォルトOFF）")
    parser.add_argument("--pace", action="store_true",
                        help="実験的な展開特徴を使う（デフォルトOFF）")
    parser.add_argument("--race-code", help="予測する単一レースの10桁URLコード")
    parser.add_argument("--output-dir", default=OUT_DIR, help="予測CSVの出力先")
    parser.add_argument("--no-lgb", action="store_true",
                        help="LightGBM アンサンブルを使わない（XGBoost 単体）")
    parser.add_argument("--no-pace", action="store_true",
                        help="展開（ペース）精緻化特徴を使わない")
    parser.add_argument("--no-mt", action="store_true",
                        help="1着分類器ブレンドを使わない（互換オプション）")
    args = parser.parse_args()

    if args.race_code is not None and not re.fullmatch(r"[0-9]{10}", args.race_code):
        parser.error("--race-code は10桁のURLコードを指定してください")
    DROP_CAREER = bool(args.drop_career)
    USE_PACE_FEATURES = bool(args.pace and not args.no_pace)

    try:
        major = int(str(xgb.__version__).split(".")[0])
        if major < 2:
            print(f"  [警告] xgboost {xgb.__version__} は rank:pairwise の lambdarank_* パラメータ未対応の可能性があります。"
                  f" 2.0 以上を推奨します。")
    except Exception:
        pass

    print("=" * 60)
    print("学習デバイスを検査中 ...")
    DEVICE = detect_device()
    print(f"  使用デバイス: {DEVICE.upper()}")

    if args.tune:
        run_tuning(n_trials=args.trials)
    elif args.predict_only:
        print("保存済みモデルを読み込み中 ...")
        (model, cat_maps, feature_cols, cat_cols, personnel_stats,
         temperature, lgb_model, ens_w, mt_model, mt_w) = load_models()
        extras = []
        if lgb_model is not None:
            extras.append(f"LightGBM重み: {ens_w:.2f}")
        if mt_model is not None:
            extras.append(f"1着分類器ブレンド重み: {mt_w:.2f}")
        print(f"  モデル読込完了（特徴量数: {len(feature_cols)}, 温度: {temperature:.3f}"
              f"{', ' + ', '.join(extras) if extras else ''}）")
        run_prediction(model, cat_maps, feature_cols, cat_cols, personnel_stats,
                       temperature, lgb_model, ens_w, mt_model, mt_w,
                       race_code=args.race_code, output_dir=args.output_dir)
    elif args.train:
        run_training(no_lgb=not args.lgb or args.no_lgb, no_mt=not args.win_blend or args.no_mt)
        print("\n" + "=" * 60)
        print("学習が完了しました。予測するには:  python predict_model.py --predict-only")
    else:
        (model, cat_maps, feature_cols, cat_cols, personnel_stats,
         temperature, lgb_model, ens_w, mt_model, mt_w) = run_training(
             no_lgb=not args.lgb or args.no_lgb, no_mt=not args.win_blend or args.no_mt)
        run_prediction(model, cat_maps, feature_cols, cat_cols, personnel_stats,
                       temperature, lgb_model, ens_w, mt_model, mt_w,
                       race_code=args.race_code, output_dir=args.output_dir)


if __name__ == "__main__":
    main()