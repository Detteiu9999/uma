# -*- coding: utf-8 -*-
"""
fetch_odds.py
JRA公式サイト（JRADB）からオッズを取得し、
  1) predictions/pred_*.csv（レースごとの予想CSV）に「単勝オッズ」列を追記して上書き保存
  2) odds/ フォルダに、レースごとの全購入方式（単勝・複勝・枠連・馬連・ワイド・馬単・3連複・3連単）
     のオッズをまとめた CSV を保存
する単独スクリプト。

JRADB は同一URL (accessO.html) 内で遷移する方式のため、
サイト内の doAction() リンクを辿って「開催選択 → レース選択 → 各オッズページ」へ遷移する。

オッズは時間ごとに変化するため、本スクリプトは複数回実行される前提で、
実行のたびに常に最新の情報で上書き保存する。

--from-predict モード:
    CSV_predict/ 内の horse_racing_data_yyppkkddrr.csv に対応するレースのオッズを
    「過去のレース結果」(accessS.html) 経由で取得する。
    過去レースの場合は確定した最終オッズが取得できる。
    各 CSV に「単勝オッズ」「複勝オッズ下限」「複勝オッズ上限」「オッズ取得時刻」列を追記し、
    odds/ フォルダに全式別のオッズ CSV を保存する（既存ファイルはスキップ）。

使い方:
    python fetch_odds.py                 # 今日の開催を対象
    python fetch_odds.py --date 20260906 # 日付を明示
    python fetch_odds.py --from-predict  # CSV_predict 内の全レースのオッズを取得
"""

import argparse
import csv
import glob
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ============================================================
# 設定
# ============================================================

JRA_ACCESS_URL = "https://www.jra.go.jp/JRADB/accessO.html"
JRA_RESULT_URL = "https://www.jra.go.jp/JRADB/accessS.html"
TOP_PAGE_CNAME = "pw15oli00/6D"  # オッズ開催選択ページ（トップページの「オッズ」リンクと同じ）
PAST_SEARCH_CNAME = "pw01skl00999999/B3"  # 過去のレース結果検索ページ（トップページの「過去のレース結果」リンクと同じ）

REQUEST_INTERVAL_SECONDS = 0.25   # サーバー負荷軽減のためのアクセス間隔
REQUEST_TIMEOUT_SECONDS = 20

# 式別名 → (ページ内リンクの表示テキスト, オッズCSVのファイル名用キー)
BET_TYPES = [
    ("単勝複勝", "tanpuku"),
    ("枠連",     "wakuren"),
    ("馬連",     "umaren"),
    ("ワイド",   "wide"),
    ("馬単",     "umatan"),
    ("3連複",    "fuku3"),
    ("3連単",    "tan3"),
]

# 枠番 → 枠色（枠連CSVの「枠色」列用）
WAKU_COLORS = {1: "白", 2: "黒", 3: "赤", 4: "青", 5: "黄", 6: "緑", 7: "橙", 8: "桃"}

# 競馬場コード → 場名（JRADBの開催選択ページの表記に合わせる）
PLACE_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}


# ============================================================
# JRADB アクセス基盤
# ============================================================

class JraOddsClient:
    """accessO.html への POST (cname=...) で遷移を再現するクライアント。"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        })
        self.last_request_time = 0.0

    def fetch(self, cname, url=JRA_ACCESS_URL):
        """cname を指定して JRADB に POST し、BeautifulSoup を返す。"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_INTERVAL_SECONDS:
            time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
        res = self.session.post(
            url, data={"cname": cname}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        self.last_request_time = time.time()
        res.raise_for_status()
        res.encoding = "shift_jis"
        html = res.text
        if "パラメータエラー" in html:
            raise RuntimeError(f"JRA側でパラメータエラーになりました (cname={cname})")
        return BeautifulSoup(html, "html.parser")


def extract_cname(onclick):
    """onclick属性の doAction('/JRADB/accessO.html', 'XXXX/YY') から cname を取り出す。"""
    m = re.search(r"doAction\(\s*'[^']*accessO\.html'\s*,\s*'([^']+)'", onclick or "")
    return m.group(1) if m else None


def parse_cname_body(cname):
    """
    cname 本体（スラッシュ前）から情報を取り出す。
    形式: pw<式別2桁><1桁><英数字2桁><区切り+場コード(計4文字)><年4桁><回2桁><日2桁>[<レース2桁>]<日付8桁>
    例: pw15orl10012026020620260906      -> 開催: 場=01, 2026年, 回=02, 日=06, 日付=20260906 (旧形式)
        pw15orl00062026040320260912      -> 開催: 場=06, 2026年, 回=04, 日=03, 日付=20260912 (直近ページ新形式)
        pw151ou1001202602060120260906Z   -> 式別=151(単勝複勝), 場=01, 回=02, 日=06, レース=01 (旧形式)
        pw151ouS306202604030120260912Z   -> 式別=151(単勝複勝), 場=06, 回=04, 日=03, レース=01 (新形式)
    ※場コード部分の表記はページ世代によって異なる:
       旧形式: '1' + 場3桁（例 "1001"）
       新形式（開催リンク）: '0' + 場3桁（例 "0006"）
       新形式（レースリンク）: "S3" + 場2桁（例 "S306"）
    """
    body = cname.split("/")[0]
    m = re.match(
        r"^pw(\d{2})([a-z0-9])[a-z0-9]{2}(.{4})(\d{4})(\d{2})(\d{2})(?:(\d{2}))?(\d{8})[A-Z]?\d*$",
        body,
    )
    if not m:
        return None
    mid = m.group(3)
    if mid[0] in "01" and mid[1:].isdigit():
        place = int(mid[1:])       # "1001" / "0006" -> 場3桁
    elif mid[2:].isdigit():
        place = int(mid[2:])       # "S306" -> 場2桁
    else:
        return None
    return {
        "bet_code": m.group(1) + m.group(2),
        "place": place,
        "year": int(m.group(4)),
        "kai": int(m.group(5)),
        "day": int(m.group(6)),
        "race": int(m.group(7)) if m.group(7) else None,
        "date": m.group(8),
    }


# ============================================================
# 過去レース（accessS.html）からのオッズ取得
# ============================================================

# 式別 → (オッズページの cname 式別コード, オッズCSVのファイル名用キー)
PAST_BET_TYPES = [
    ("151", "tanpuku"),
    ("153", "wakuren"),
    ("154", "umaren"),
    ("155", "wide"),
    ("156", "umatan"),
    ("157", "fuku3"),
    ("158", "tan3"),
]


def parse_srl_cname(cname):
    """
    開催ページの cname から情報を取り出す。
    例: pw01srl10062026010120260104/24 -> 場=06, 2026年, 回=01, 日=01, 日付=20260104
    """
    m = re.match(r"^pw01srl1(\d{3})(\d{4})(\d{2})(\d{2})(\d{8})", cname.split("/")[0])
    if not m:
        return None
    return {
        "place": int(m.group(1)),
        "year": int(m.group(2)),
        "kai": int(m.group(3)),
        "day": int(m.group(4)),
        "date": m.group(5),
    }


def find_past_month_page(client, year, month):
    """
    過去のレース結果検索ページで使われている objParam テーブルから
    指定年月の月ページ cname を取得する。
    （検索ページの「表示」ボタンと同じ仕組み: cname = pw01skl10<YYMM>/<objParam値>）
    """
    soup = client.fetch(PAST_SEARCH_CNAME, url=JRA_RESULT_URL)
    html = str(soup)
    m = re.search(r'var yearMonth = "(\d{6})"', html)
    threshold = int(m.group(1)) if m else 999999
    yy = year % 100
    key = f"{yy:02d}{month:02d}"
    m = re.search(r'objParam\["' + key + r'"\]="([0-9A-Z]{2})"', html)
    if not m:
        return None
    prefix = "pw01skl00" if (year * 100 + month) >= threshold else "pw01skl10"
    return f"{prefix}{year}{month:02d}/{m.group(1)}"


def find_kaisai_cname(client, year, month, place, kai, day):
    """月ページから、指定の競馬場・回・日の開催ページ (pw01srl...) の cname を探す。"""
    month_cname = find_past_month_page(client, year, month)
    if not month_cname:
        return None
    soup = client.fetch(month_cname, url=JRA_RESULT_URL)
    for m in re.finditer(
            r"doAction\('[^']*accessS\.html',\s*'(pw01srl[^']+)'\)", str(soup)):
        info = parse_srl_cname(m.group(1))
        if info and (info["place"], info["kai"], info["day"]) == (place, kai, day):
            return m.group(1)
    return None


def find_past_race_links(client, kaisai_cname):
    """
    開催ページ（レース選択ページ）から、レース番号ごとの
    「最終オッズ」リンク（単勝複勝ページ）を集める。
    戻り値: {レース番号: 単勝複勝ページの cname}
    """
    soup = client.fetch(kaisai_cname, url=JRA_RESULT_URL)
    result = {}
    for a in soup.find_all("a", onclick=True):
        m = re.search(r"doAction\('[^']*accessO\.html',\s*'(pw151ou[^']+)'\)", a["onclick"])
        if not m:
            continue
        info = parse_cname_body(m.group(1))
        if info and info["race"]:
            result[info["race"]] = m.group(1)
    return result


def find_bet_type_cnames(client, tanpuku_cname):
    """
    単勝複勝オッズページ内の式別タブから、各式別ページの cname を集める。
    戻り値: {式別コード: cname}
    """
    soup = client.fetch(tanpuku_cname)
    result = {}
    for a in soup.find_all("a", onclick=True):
        m = re.search(r"doAction\('[^']*accessO\.html',\s*'(pw(15[3-8])ou[^']+)'\)", a["onclick"])
        if m:
            result[m.group(2)] = m.group(1)
    return result


def parse_past_tan3(soup):
    """
    3連単ページ（馬番順）の解析。
    div.tan3_unit ごとに「1着馬」、その中の各 table の前にある div.p_line の
    「2着 N」が2着馬、各行の <th> が3着馬、<td> がオッズ。
    → [(n1, n2, n3, odds_text), ...]
    """
    rows = []
    for unit in soup.find_all("div", class_="tan3_unit"):
        header = unit.find("h4", class_="sub_header")
        if not header:
            continue
        num_el = header.find("span", class_="num")
        if not num_el:
            continue
        try:
            first = int(_text(num_el))
        except ValueError:
            continue
        for table in unit.find_all("table", class_="tan3"):
            # 直前の p_line 群から "2着 N" を探す
            second = None
            for prev in table.find_all_previous("div", class_="p_line"):
                cap = prev.find("div", class_="cap")
                num = prev.find("div", class_="num")
                if cap and num and "2着" in cap.get_text():
                    try:
                        second = int(num.get_text(strip=True))
                    except ValueError:
                        second = None
                    break
                if cap and "1着" in cap.get_text():
                    break
            if second is None:
                continue
            for tr in table.find_all("tr"):
                th = tr.find("th")
                td = tr.find("td")
                if not th or not td:
                    continue
                try:
                    third = int(_text(th))
                except ValueError:
                    continue
                odds = _text(td)
                if odds:
                    rows.append((first, second, third, odds))
    return rows


def fetch_past_race_odds(client, tanpuku_cname):
    """
    過去レースの最終オッズを全式別分取得する。
    戻り値: {式別キー: DataFrame}（build_odds_dataframes と同じ列構成）
    """
    tanpuku = parse_tanpuku(client.fetch(tanpuku_cname))
    bet_cnames = find_bet_type_cnames(client, tanpuku_cname)

    pairs = {}
    fuku3 = []
    tan3 = []
    for code, key in PAST_BET_TYPES:
        if code == "151":
            continue
        cn = bet_cnames.get(code)
        if not cn:
            continue
        soup = client.fetch(cn)
        if code == "153":
            pairs["枠連"] = parse_pair_tables(soup, "waku")
        elif code == "154":
            pairs["馬連"] = parse_pair_tables(soup, "umaren")
        elif code == "155":
            pairs["ワイド"] = parse_pair_tables(soup, "wide")
        elif code == "156":
            pairs["馬単"] = parse_pair_tables(soup, "umatan")
        elif code == "157":
            fuku3 = parse_fuku3(soup)
        elif code == "158":
            tan3 = parse_past_tan3(soup)
    return tanpuku, pairs, fuku3, tan3


# ============================================================
# 各オッズページの解析
# ============================================================

def _text(cell):
    return cell.get_text(strip=True)


def parse_tanpuku(soup):
    """単勝・複勝ページ → {馬番: {...}}（枠番は rowspan 結合を考慮して前行から引き継ぐ）"""
    table = soup.find("table", class_="tanpuku")
    result = {}
    if not table:
        return result
    last_waku = ""
    for tr in table.find_all("tr"):
        num_td = tr.find("td", class_="num")
        if not num_td:
            continue
        try:
            umaban = int(_text(num_td))
        except ValueError:
            continue
        tan_td = tr.find("td", class_="odds_tan")
        fuku_td = tr.find("td", class_="odds_fuku")
        horse_td = tr.find("td", class_="horse")
        waku_img = tr.find("td", class_="waku")
        if waku_img:
            img = waku_img.find("img")
            if img and img.get("src"):
                m = re.search(r"/(\d+)\.png", img["src"])
                if m:
                    last_waku = m.group(1)
        waku = last_waku
        fuku_min = fuku_max = ""
        if fuku_td:
            min_el = fuku_td.find("span", class_="min")
            max_el = fuku_td.find("span", class_="max")
            fuku_min = _text(min_el) if min_el else ""
            fuku_max = _text(max_el) if max_el else ""
        result[umaban] = {
            "枠番": waku,
            "馬番": umaban,
            "馬名": _text(horse_td) if horse_td else "",
            "単勝オッズ": _text(tan_td) if tan_td else "",
            "複勝オッズ下限": fuku_min,
            "複勝オッズ上限": fuku_max,
        }
    return result


def parse_pair_tables(soup, table_class):
    """
    枠連・馬連・ワイド・馬単ページの解析。
    各 table の <caption> が1軸目の番号、各行の <th> が2軸目の番号、<td> がオッズ。
    → [(first, second, odds_text), ...]
    """
    rows = []
    for table in soup.find_all("table", class_=table_class):
        caption = table.find("caption")
        if not caption:
            continue
        if table_class == "waku":
            # 枠連は caption が画像（alt="枠1白" など）
            img = caption.find("img")
            m = re.search(r"枠(\d+)", img.get("alt", "")) if img else None
            if not m:
                continue
            first = int(m.group(1))
        else:
            try:
                first = int(_text(caption))
            except ValueError:
                continue
        for tr in table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            try:
                second = int(_text(th))
            except ValueError:
                continue
            odds = _text(td)
            if odds:
                rows.append((first, second, odds))
    return rows


def parse_fuku3(soup):
    """
    3連複ページの解析。
    各 table の <caption> が "1-2" のような2頭の組、各行の <th> が3頭目、<td> がオッズ。
    → [(n1, n2, n3, odds_text), ...]（馬番は昇順に正規化）
    """
    rows = []
    for table in soup.find_all("table", class_="fuku3"):
        caption = table.find("caption")
        if not caption:
            continue
        m = re.match(r"^(\d+)-(\d+)$", _text(caption))
        if not m:
            continue
        n1, n2 = int(m.group(1)), int(m.group(2))
        for tr in table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            try:
                n3 = int(_text(th))
            except ValueError:
                continue
            odds = _text(td)
            if odds:
                rows.append((n1, n2, n3, odds))
    return rows


def parse_tan3(soup):
    """
    3連単ページの解析。
    div.tan3_unit ごとに「1着馬」、その中の各 table 直前の div.p_line に「2着N」、
    各行の <th> が3着馬、<td> がオッズ。
    → [(n1, n2, n3, odds_text), ...]
    """
    rows = []
    for unit in soup.find_all("div", class_="tan3_unit"):
        header = unit.find("h4", class_="sub_header")
        if not header:
            continue
        num_el = header.find("span", class_="num")
        if not num_el:
            # "1ルージュエピック" のようなテキスト先頭から馬番を取る
            m = re.match(r"\s*(\d+)", _text(header))
            if not m:
                continue
            first = int(m.group(1))
        else:
            first = int(_text(num_el))
        for table in unit.find_all("table", class_="tan3"):
            # 直前の p_line に "2着N" がある
            second = None
            for prev in table.find_all_previous("div", class_="p_line"):
                txt = _text(prev)
                m = re.match(r"^2着\s*(\d+)$", txt)
                if m:
                    second = int(m.group(1))
                    break
                if txt.startswith("1着"):
                    break
            if second is None:
                continue
            for tr in table.find_all("tr"):
                th = tr.find("th")
                td = tr.find("td")
                if not th or not td:
                    continue
                try:
                    third = int(_text(th))
                except ValueError:
                    continue
                odds = _text(td)
                if odds:
                    rows.append((first, second, third, odds))
    return rows


# ============================================================
# 開催・レースの特定
# ============================================================


def find_race_cnames(soup):
    """
    レース選択ページから、式別ごとの {レース番号: cname} を集める。
    戻り値: {式別表示名: {レース番号: cname}}
    """
    result = {name: {} for name, _ in BET_TYPES}
    for a in soup.find_all("a", onclick=True):
        cname = extract_cname(a.get("onclick"))
        if not cname:
            continue
        info = parse_cname_body(cname)
        if not info:
            continue
        label = a.get_text(strip=True)
        for name, _ in BET_TYPES:
            if label == name:
                result[name][info["race"]] = cname
    return result


# ============================================================
# CSV 出力
# ============================================================

def build_odds_dataframes(race_num, tanpuku, pairs, fuku3, tan3):
    """取得したオッズから、式別ごとの DataFrame を作る。"""
    dfs = {}

    # 単勝・複勝
    if tanpuku:
        df = pd.DataFrame([tanpuku[k] for k in sorted(tanpuku)])
        df.insert(0, "レース番号", f"{race_num}R")
        dfs["tanpuku"] = df

    # 枠連
    if pairs.get("枠連"):
        rows = []
        for w1, w2, odds in pairs["枠連"]:
            rows.append({
                "レース番号": f"{race_num}R",
                "枠番1": w1, "枠番2": w2,
                "枠色1": WAKU_COLORS.get(w1, ""), "枠色2": WAKU_COLORS.get(w2, ""),
                "枠連オッズ": odds,
            })
        dfs["wakuren"] = pd.DataFrame(rows)

    # 馬連・ワイド・馬単
    for name, key, col in [("馬連", "umaren", "馬連オッズ"),
                           ("ワイド", "wide", "ワイドオッズ"),
                           ("馬単", "umatan", "馬単オッズ")]:
        if pairs.get(name):
            rows = [{"レース番号": f"{race_num}R", "馬番1": a, "馬番2": b, col: o}
                    for a, b, o in pairs[name]]
            dfs[key] = pd.DataFrame(rows)

    # 3連複
    if fuku3:
        rows = [{"レース番号": f"{race_num}R", "馬番1": a, "馬番2": b, "馬番3": c,
                 "3連複オッズ": o} for a, b, c, o in fuku3]
        dfs["fuku3"] = pd.DataFrame(rows)

    # 3連単
    if tan3:
        rows = [{"レース番号": f"{race_num}R", "1着馬番": a, "2着馬番": b, "3着馬番": c,
                 "3連単オッズ": o} for a, b, c, o in tan3]
        dfs["tan3"] = pd.DataFrame(rows)

    return dfs


def save_odds_csvs(dfs, odds_dir, date_str, place_name, kai, day, race_num):
    """式別ごとの DataFrame を odds/ に上書き保存する。"""
    base = f"odds_{date_str}_{place_name}{kai}回{day}日_{race_num}R"
    saved = []
    for key, df in dfs.items():
        path = os.path.join(odds_dir, f"{base}_{key}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        saved.append(path)
    return saved


def update_prediction_csvs(pred_dir, race_keys, tanpuku, fetched_at):
    """
    predictions/ 内の pred_*.csv のうち、開催日・競馬場・レース番号が一致するものに
    「単勝オッズ」「複勝オッズ下限」「複勝オッズ上限」「オッズ取得時刻」を追記して上書き保存する。
    戻り値: 更新したファイルパスのリスト
    """
    updated = []
    for path in glob.glob(os.path.join(pred_dir, "pred_*.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if not {"開催日", "競馬場", "レース番号", "馬番"} <= set(df.columns):
            continue
        # このファイルが対象レースか判定（全行同一レース前提）
        key = (str(df["開催日"].iloc[0]), str(df["競馬場"].iloc[0]), str(df["レース番号"].iloc[0]))
        if key not in race_keys:
            continue
        def lookup(umaban, field):
            try:
                rec = tanpuku.get(int(umaban))
            except (ValueError, TypeError):
                return ""
            return rec.get(field, "") if rec else ""
        df["単勝オッズ"] = [lookup(u, "単勝オッズ") for u in df["馬番"]]
        df["複勝オッズ下限"] = [lookup(u, "複勝オッズ下限") for u in df["馬番"]]
        df["複勝オッズ上限"] = [lookup(u, "複勝オッズ上限") for u in df["馬番"]]
        df["オッズ取得時刻"] = fetched_at
        df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        updated.append(path)
    return updated


# ============================================================
# メイン処理（CSV_predict 内の過去レース）
# ============================================================

def update_predict_csv(path, tanpuku, fetched_at):
    """
    CSV_predict 内の horse_racing_data_*.csv に
    「単勝オッズ」「複勝オッズ下限」「複勝オッズ上限」「オッズ取得時刻」を追記して上書き保存する。
    馬番が一致しない行は馬名でマッチングする。
    戻り値: 更新したかどうか
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "馬番" not in df.columns:
        return False

    by_num = tanpuku
    by_name = {rec["馬名"]: rec for rec in tanpuku.values() if rec.get("馬名")}

    def lookup(row, field):
        try:
            rec = by_num.get(int(row["馬番"]))
        except (ValueError, TypeError):
            rec = None
        if rec is None and "馬名" in df.columns:
            rec = by_name.get(str(row["馬名"]).strip())
        return rec.get(field, "") if rec else ""

    df["単勝オッズ"] = [lookup(row, "単勝オッズ") for _, row in df.iterrows()]
    df["複勝オッズ下限"] = [lookup(row, "複勝オッズ下限") for _, row in df.iterrows()]
    df["複勝オッズ上限"] = [lookup(row, "複勝オッズ上限") for _, row in df.iterrows()]
    df["オッズ取得時刻"] = fetched_at
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return True


def main_from_predict(args):
    """CSV_predict/ 内の全レースについて、過去のレース結果ページから最終オッズを取得する。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    predict_dir = os.path.join(base_dir, "CSV_predict")
    odds_dir = os.path.join(base_dir, "odds")
    os.makedirs(odds_dir, exist_ok=True)

    # 対象レースをファイル名から収集: horse_racing_data_yyppkkddrr.csv
    targets = {}  # (year, place, kai, day) -> {race_num: csv_path}
    for path in sorted(glob.glob(os.path.join(predict_dir, "horse_racing_data_*.csv"))):
        m = re.match(r"horse_racing_data_(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.csv",
                     os.path.basename(path))
        if not m:
            continue
        year = 2000 + int(m.group(1))
        place, kai, day, race_num = (int(m.group(2)), int(m.group(3)),
                                     int(m.group(4)), int(m.group(5)))
        if args.place and place != args.place:
            continue
        if args.race and race_num != args.race:
            continue
        targets.setdefault((year, place, kai, day), {})[race_num] = path

    if not targets:
        print("[対象なし] CSV_predict/ に horse_racing_data_*.csv が見つかりませんでした。")
        return

    total_races = sum(len(v) for v in targets.values())
    print(f"=== 過去レースのオッズ取得: {len(targets)}開催 / {total_races}レース ===")

    client = JraOddsClient()
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kaisai_cache = {}  # (year, month, place, kai, day) -> 開催ページ cname（探索済みは再アクセスしない）

    for (year, place, kai, day), races in sorted(targets.items()):
        place_name = PLACE_NAMES.get(place, f"場{place}")
        print(f"\n--- {year}年 {place_name} {kai}回{day}日 ({len(races)}レース) ---")

        # 開催日が属する月を特定するため、先頭レースのCSVから開催日を読む
        first_path = races[sorted(races)[0]]
        date_str = None
        try:
            df_head = pd.read_csv(first_path, encoding="utf-8-sig", nrows=1)
            for col in ("開催年月日", "開催日"):
                if col in df_head.columns:
                    dm = re.search(r"(\d{4})\D*(\d{1,2})\D*(\d{1,2})",
                                   str(df_head[col].iloc[0]))
                    if dm:
                        date_str = f"{dm.group(1)}{int(dm.group(2)):02d}{int(dm.group(3)):02d}"
                        break
        except Exception:
            pass

        # 開催ページを探索（開催日が分かればその月から、不明なら1月→12月の順）
        if date_str:
            months = [int(date_str[4:6])]
        else:
            months = list(range(1, 13))
        kaisai_cname = None
        for month in months:
            cache_key = (year, month, place, kai, day)
            if cache_key in kaisai_cache:
                kaisai_cname = kaisai_cache[cache_key]
                if kaisai_cname:
                    break
                continue
            try:
                kaisai_cname = find_kaisai_cname(client, year, month, place, kai, day)
            except Exception as e:
                print(f"[エラー] {year}年{month}月の月ページ取得に失敗: {e}")
                kaisai_cname = None
            kaisai_cache[cache_key] = kaisai_cname
            if kaisai_cname:
                break
        if not kaisai_cname:
            print("[対象なし] 開催がJRAサイトに見つかりませんでした。")
            continue

        # レース選択ページから各レースのオッズリンクを取得
        try:
            race_links = find_past_race_links(client, kaisai_cname)
        except Exception as e:
            print(f"[エラー] レース選択ページの取得に失敗: {e}")
            continue

        # 開催日を cname から取得（CSVから読めなかった場合のフォールバック）
        if not date_str:
            info = parse_srl_cname(kaisai_cname)
            date_str = info["date"] if info else f"{year}0000"

        for race_num in sorted(races):
            cn = race_links.get(race_num)
            if not cn:
                print(f"  [{race_num}R] オッズリンクが見つかりません（未発売・中止の可能性）")
                continue

            # 既に取得済み（CSV_predict に単勝オッズ列があり、odds/ にCSVがある）場合はスキップ
            tanpuku_csv = os.path.join(
                odds_dir, f"odds_{date_str}_{place_name}{kai}回{day}日_{race_num}R_tanpuku.csv")
            try:
                df_head = pd.read_csv(races[race_num], encoding="utf-8-sig", nrows=0)
                if "単勝オッズ" in df_head.columns and os.path.exists(tanpuku_csv):
                    print(f"  [{race_num}R] 既存のためスキップ")
                    continue
            except Exception:
                pass

            try:
                tanpuku, pairs, fuku3, tan3 = fetch_past_race_odds(client, cn)
            except Exception as e:
                print(f"  [{race_num}R] 取得に失敗: {e}")
                continue

            # オッズCSV保存
            dfs = build_odds_dataframes(race_num, tanpuku, pairs, fuku3, tan3)
            saved = save_odds_csvs(dfs, odds_dir, date_str, place_name, kai, day, race_num)
            counts = ", ".join(f"{k}:{len(v)}件" for k, v in dfs.items())
            print(f"  [{race_num}R] オッズCSV保存: {len(saved)}ファイル ({counts})")

            # CSV_predict への単勝オッズ追記
            if update_predict_csv(races[race_num], tanpuku, fetched_at):
                print(f"  [{race_num}R] CSV更新: {os.path.basename(races[race_num])}")

    print("\n=== 全処理が完了しました ===")


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="JRA公式サイトからオッズを取得しCSVに追記・保存する")
    parser.add_argument("--date", help="対象日 (YYYYMMDD)。省略時は今日。")
    parser.add_argument("--place", type=int, help="競馬場コード(1-10)で絞り込み")
    parser.add_argument("--race", type=int, help="レース番号で絞り込み")
    parser.add_argument("--from-predict", action="store_true",
                        help="CSV_predict/ 内の全レースを対象に、過去のレース結果から最終オッズを取得する")
    args = parser.parse_args()

    if args.from_predict:
        main_from_predict(args)
        return

    target_date = args.date or datetime.now().strftime("%Y%m%d")
    try:
        date_obj = datetime.strptime(target_date, "%Y%m%d")
    except ValueError:
        print(f"[エラー] --date の形式が不正です: {target_date} (YYYYMMDD で指定)")
        return
    date_str = date_obj.strftime("%Y%m%d")
    year = date_obj.year

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, "predictions")
    odds_dir = os.path.join(base_dir, "odds")
    os.makedirs(odds_dir, exist_ok=True)

    client = JraOddsClient()

    # --- 開催選択ページ ---
    print(f"=== オッズ取得: {date_obj.strftime('%Y年%m月%d日')} の開催を検索 ===")
    try:
        soup = client.fetch(TOP_PAGE_CNAME)
    except Exception as e:
        print(f"[エラー] 開催選択ページの取得に失敗: {e}")
        return

    # 開催リンクを列挙し、対象日のものを抽出
    kaisai_list = []  # (cname, info)
    for a in soup.find_all("a", onclick=True):
        cname = extract_cname(a.get("onclick"))
        if not cname:
            continue
        info = parse_cname_body(cname)
        if not info:
            continue
        # 開催選択ページへのリンク（レース番号を持たないもの）だけを対象にする
        if info["race"] is None and info["date"] == date_str:
            kaisai_list.append((cname, info, a.get_text(strip=True)))

    if not kaisai_list:
        print(f"[対象なし] {date_obj.strftime('%Y年%m月%d日')} の開催がJRAサイトに見つかりませんでした。")
        print("（開催日以外・またはまだオッズが公開されていない可能性があります）")
        return

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for cname, info, label in kaisai_list:
        place = info["place"]
        kai = info["kai"]
        day = info["day"]
        place_name = PLACE_NAMES.get(place, f"場{place}")
        if args.place and place != args.place:
            continue

        print(f"\n--- {label} ({place_name} {kai}回{day}日) ---")

        # --- レース選択ページ ---
        try:
            race_soup = client.fetch(cname)
        except Exception as e:
            print(f"[エラー] レース選択ページの取得に失敗: {e}")
            continue
        race_cnames = find_race_cnames(race_soup)

        races = sorted(race_cnames["単勝複勝"].keys())
        if args.race:
            races = [r for r in races if r == args.race]
        if not races:
            print("[対象なし] レースが見つかりませんでした。")
            continue

        for race_num in races:
            print(f"  [{race_num}R] オッズ取得中...")
            try:
                # 単勝複勝（予想CSV追記用・馬番マスタとしても使用）
                tanpuku = parse_tanpuku(client.fetch(race_cnames["単勝複勝"][race_num]))

                # 2連系
                pairs = {}
                for name, key in [("枠連", "waku"), ("馬連", "umaren"),
                                  ("ワイド", "wide"), ("馬単", "umatan")]:
                    cn = race_cnames.get(name, {}).get(race_num)
                    if cn:
                        pairs[name] = parse_pair_tables(client.fetch(cn), key)

                # 3連系
                fuku3 = []
                cn = race_cnames.get("3連複", {}).get(race_num)
                if cn:
                    fuku3 = parse_fuku3(client.fetch(cn))
                tan3 = []
                cn = race_cnames.get("3連単", {}).get(race_num)
                if cn:
                    tan3 = parse_tan3(client.fetch(cn))
            except Exception as e:
                print(f"  [エラー] {race_num}R の取得に失敗: {e}")
                continue

            # --- オッズCSV保存（常に上書き） ---
            dfs = build_odds_dataframes(race_num, tanpuku, pairs, fuku3, tan3)
            saved = save_odds_csvs(dfs, odds_dir, date_str, place_name, kai, day, race_num)
            counts = ", ".join(f"{k}:{len(v)}件" for k, v in dfs.items())
            print(f"  [{race_num}R] オッズCSV保存: {len(saved)}ファイル ({counts})")

            # --- 予想CSVへの単勝オッズ追記（常に上書き） ---
            race_keys = {
                (f"{year}年 第{kai}回{day}日目", place_name, f"{race_num}R"),
            }
            updated = update_prediction_csvs(pred_dir, race_keys, tanpuku, fetched_at)
            for p in updated:
                print(f"  [{race_num}R] 予想CSV更新: {os.path.basename(p)}")
            if not updated:
                print(f"  [{race_num}R] 対応する予想CSVが predictions/ に見つかりませんでした"
                      f"（開催日='{year}年 第{kai}回{day}日目', 競馬場='{place_name}', レース番号='{race_num}R'）")

    print("\n=== 全処理が完了しました ===")


if __name__ == "__main__":
    main()
