import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
import unicodedata
from datetime import date

# ============================================================
# 設定
# ============================================================

TARGET_YEARS = [26]
PLACES = [6,9]
KAIS = [4]
DAYS = [5]
RACES = range(1, 13)

BASE_URL = "https://jiro8.sakura.ne.jp/index.php?code="
CSV_PREFIX = "horse_racing_data"

# 既存CSVに今回追加する列がなければ、そのレースを再取得するか
RESCRAPE_IF_COLUMNS_MISSING = True

REQUEST_INTERVAL_SECONDS = 0.5


# ============================================================
# データ分割用補助関数
# ============================================================

def split_course_info(value):
    """ "ダ1800不", "芝2000稍" などを 芝/ダート, 距離, 馬場 に分割 """
    text = str(value).strip()
    match = re.search(r"^(芝|ダ|障|ダート)?\s*(\d{3,4})\s*(良|稍重|稍|重|不良|不)?", text)
    if match:
        track = match.group(1) or ""
        dist = match.group(2) or ""
        cond = match.group(3) or ""
        if track == "ダ": track = "ダート"
        if cond == "稍": cond = "稍重"
        if cond == "不": cond = "不良"
        return track, dist, cond
    return text, "", ""

def split_time_and_finish(value):
    """ "1.53.4④" などを タイム, 着順 に分割 """
    text = str(value).strip()
    if re.search(r"(中止|取消|除外)", text):
        return text, ""
    
    parts = text.split()
    if len(parts) >= 2:
        race_time = parts[0]
        finish = "".join(parts[1:])
    else:
        text_nospace = re.sub(r"\s+", "", text)
        time_match = re.search(r"([0-9]+(?:[.:][0-9]+){1,2})", text_nospace)
        if time_match:
            race_time = time_match.group(1)
            finish = text_nospace[time_match.end():]
        else:
            race_time = text_nospace
            finish = ""

    race_time = unicodedata.normalize("NFKC", race_time)
    finish = unicodedata.normalize("NFKC", finish)
    finish_match = re.search(r"([0-9]+)", finish)
    finish = finish_match.group(1) if finish_match else ""
    return race_time, finish

def calc_days_between(current_date, past_md):
    """ 本走日(datetime.date) と 前走の "MM/DD" 文字列から経過日数を計算 """
    if current_date is None:
        return ""
    text = str(past_md).strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})$", text)
    if not m:
        return ""
    month = int(m.group(1))
    day = int(m.group(2))
    year = current_date.year
    try:
        past_date = date(year, month, day)
    except ValueError:
        return ""
    # 前走が本走より後の日付になる場合は前年とみなす
    if past_date > current_date:
        try:
            past_date = date(year - 1, month, day)
        except ValueError:
            return ""
    return (current_date - past_date).days

def split_date_place_weather(value):
    """ "08/02中曇" などを 日付, 競馬場, 天候 に分割 """
    text = str(value).strip()
    match = re.search(r'^(\d{1,2}/\d{1,2})([^\d晴曇雨雪]+?)?(小雨|小雪|晴|曇|雨|小|雪)?$', text)
    if match:
        d = match.group(1) or ""
        p = match.group(2) or ""
        w = match.group(3) or ""
        return d, p, w
    return text, "", ""

def split_jockey_weight(value):
    """ "J.コレ57" などを 騎手, 斤量 に分割 """
    text = str(value).strip()
    match = re.search(r'^(.+?)([\d.]+)$', text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, ""

def split_head_num_pop(value):
    """ "15ﾄ4番13" などを 頭数, 馬番, 人気 に分割 """
    text = str(value).strip()
    match = re.search(r'(\d+)\D+(\d+)\D+(\d+)', text)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "", "", ""

def split_pace_leg_3f(value):
    """ "M追36.1" などを ペース, 脚質, 上3F に分割 """
    text = str(value).strip()
    match = re.search(r'^([A-Za-z]+)?([^A-Za-z\d.]+)?([\d.]+)?$', text)
    if match:
        p = match.group(1) or ""
        l = match.group(2) or ""
        t = match.group(3) or ""
        return p, l, t
    return "", "", ""

def split_top_margin(value):
    """ "ﾓﾓﾝｳ(0.4)" などを トップ馬名, タイム差 に分割 """
    text = str(value).strip()
    match = re.search(r'^(.*?)\((.*?)\)$', text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, ""

def split_weight_change_3f(value):
    """ "554(+4)1" などを 馬体重, 増減, 上がり3F順位 に分割 """
    text = str(value).strip()
    match = re.search(r'^([^\(]+)(?:\((.*?)\))?(\d*)$', text)
    if match:
        w = match.group(1).strip()
        c = match.group(2) or ""
        r = match.group(3) or ""
        return w, c, r
    return text, "", ""


def normalize_horse_name(text):
    text = str(text)
    text = text.replace("ｌ", "ー").replace("｜", "ー").replace("￨", "ー").replace("¦", "ー")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[(（](?:地|外|父|市|招|特)[)）]", "", text)
    # カタカナ名に紛れ込んだ半角英字 "l"（長音の誤記）を "ー" に補正。
    # 例: シケlダ → シケーダ（Invincible 等の英馬名中の l はカタカナと隣接しないため影響しない）
    text = re.sub(r"(?<=[ァ-ヶー])l(?=[ァ-ヶー]|$)", "ー", text)
    text = text.replace("bーe", "ble")
    return text

def extract_cell_text(node):
    """縦書きセル内の <br> で区切られた文字を結合して取り出す"""
    html_str = str(node)
    html_str = re.sub(r'<br\s*/?>', '', html_str, flags=re.IGNORECASE)
    temp_soup = BeautifulSoup(html_str, "html.parser")
    return temp_soup.get_text(separator="", strip=True)


def extract_horse_name(cell):
    target_node = None
    nested_table = cell.find("table")
    if nested_table:
        first_tr = nested_table.find("tr")
        if first_tr:
            tds = first_tr.find_all("td")
            if len(tds) >= 2:
                target_node = tds[1]  
            elif len(tds) == 1:
                target_node = tds[0]

    if not target_node:
        target_node = cell

    raw_text = extract_cell_text(target_node)
    
    name = normalize_horse_name(raw_text)
    if "／" in name or "/" in name:
        parts = re.split(r'[／/]', name)
        for part in parts:
            part = re.sub(r"^[(（].*?[)）]", "", part)
            if 2 <= len(part) <= 9:
                return part
    return name


def extract_pedigree(cell):
    """馬名セルから 馬名, 父馬名, 母馬名, 母父馬名 を抽出する。

    馬名セル内のネスト表は縦書き3列構成で、
        左列(c232, rowspan=2): 母馬名 ／ 母父馬名
        中列(c231, rowspan=2): 馬名
        右列(c232)           : 父馬名
    となっている。各列は <br> で1文字ずつ分かれているため結合してから正規化する。
    （全角アルファベットは normalize_horse_name 内の NFKC で半角に統一される）
    """
    sire, dam, damsire = "", "", ""
    nested_table = cell.find("table")
    if nested_table:
        first_tr = nested_table.find("tr")
        if first_tr:
            tds = first_tr.find_all("td")
            if len(tds) >= 3:
                dam_text = normalize_horse_name(extract_cell_text(tds[0]))
                parts = re.split(r'[／/]', dam_text)
                dam = parts[0].strip() if parts else ""
                damsire = parts[1].strip() if len(parts) >= 2 else ""
                sire = normalize_horse_name(extract_cell_text(tds[2]))
    name = extract_horse_name(cell)
    return name, sire, dam, damsire


# ============================================================
# HTML解析
# ============================================================

def parse_race_html(html_content, url_code):
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", class_="c1")
    if not table:
        return None

    # --- レース基本情報の抽出 ---
    track_type = ""
    distance = ""
    direction = ""
    track_condition = ""
    
    nobr_tag = soup.find("nobr")
    header_text = nobr_tag.get_text(separator=" ", strip=True) if nobr_tag else soup.get_text(separator=" ")

    course_match = re.search(r'(芝|ダート|障害)[・\s]*([^\s0-9a-zA-Z]+)?\s*(\d{3,4})m', header_text)
    if course_match:
        track_type = course_match.group(1)
        direction = (course_match.group(2) or "").replace("・", "").strip()
        distance = course_match.group(3)

    condition_match = re.search(r'(良|稍重|重|不良)', header_text)
    if condition_match:
        track_condition = condition_match.group(1)

    # --- 本走の開催日を抽出 ---
    race_date = None
    date_match = re.search(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', header_text)
    if date_match:
        try:
            race_date = date(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
        except ValueError:
            race_date = None

    # --- 表記の正規化 ---
    def normalize_label(text):
        text = unicodedata.normalize("NFKC", str(text))
        text = re.sub(r"\s+", "", text)
        label_map = {
            "ﾍﾟｰｽ,脚質,上3F": "ペース,脚質,上3F",
            "ﾀｲﾑ,(着順)": "タイム,(着順)",
            "ﾄｯﾌﾟ(ﾀｲﾑ差)": "トップ(タイム差)",
        }
        return label_map.get(text, text)

    raw_rows = []
    for tr in table.find_all("tr", recursive=False):
        html_cells = tr.find_all(["td", "th"], recursive=False)
        if not html_cells:
            continue
        cells = [cell.get_text(separator=" ", strip=True) for cell in html_cells]
        raw_rows.append({"cells": cells, "html_cells": html_cells})

    num_horses = 0
    for row in raw_rows:
        cells = row["cells"]
        if len(cells) >= 2 and normalize_label(cells[-1]) == "馬番":
            num_horses = len(cells) - 1
            break
    if not num_horses:
        return None

    current_race_fields = [
        "着順", "単勝オッズ", "タイム", "ペース,脚質,上3F", 
        "通過順位", "馬体重()3F順", "先行指数", "ペース指数", 
        "上がり指数", "スピード指数"
    ]
    past_race_fields = [
        "成績", "レース名", "コース", "騎手,斤量", "頭数,馬番,人気", 
        "タイム,(着順)", "ペース,脚質,上3F", "通過順位", "トップ(タイム差)", 
        "馬体重()3F順", "先行指数", "ペース指数", "上がり指数", "スピード指数"
    ]

    data_dict = {}
    current_race_field_index = None
    current_past_num = None
    current_past_field_index = None

    for row in raw_rows:
        cells = row["cells"]
        html_cells = row["html_cells"]
        row_values = cells[:num_horses]
        row_title = normalize_label(cells[-1]) if len(cells) > num_horses else ""

        if row_title == "馬名":
            pedigrees = [extract_pedigree(cell) for cell in html_cells[:num_horses]]
            data_dict["馬名"] = [p[0] for p in pedigrees]
            data_dict["父馬"] = [p[1] for p in pedigrees]
            data_dict["母馬"] = [p[2] for p in pedigrees]
            data_dict["母父馬"] = [p[3] for p in pedigrees]
            continue

        # --- 当レースの解析 (最新結果) ---
        if current_race_field_index is not None:
            field_name = current_race_fields[current_race_field_index]
            
            if field_name == "ペース,脚質,上3F":
                paces, legs, times = [], [], []
                for val in row_values:
                    p, l, t = split_pace_leg_3f(val)
                    paces.append(p); legs.append(l); times.append(t)
                data_dict["ペース"] = paces
                data_dict["脚質"] = legs
                data_dict["上3F"] = times
            elif field_name == "馬体重()3F順":
                weights, changes, ranks = [], [], []
                for val in row_values:
                    w, c, r = split_weight_change_3f(val)
                    weights.append(w); changes.append(c); ranks.append(r)
                data_dict["馬体重"] = weights
                data_dict["体重増減"] = changes
                data_dict["上がり3F順位"] = ranks
            else:
                data_dict[field_name] = row_values
            
            current_race_field_index += 1
            if current_race_field_index >= len(current_race_fields):
                current_race_field_index = None
            continue

        # --- 過去レースの開始位置検出 ---
        match = re.fullmatch(r"(前走|[2-5]走前)の成績", row_title)
        if match:
            past_label = match.group(1)
            current_past_num = 1 if past_label == "前走" else int(re.search(r"\d+", past_label).group())
            prefix = "前走" if current_past_num == 1 else f"{current_past_num}走前"

            # このマーカー行には「成績」(日付・競馬場・天候) のデータが含まれる
            dates, places, weathers, intervals = [], [], [], []
            for val in row_values:
                d, p, w = split_date_place_weather(val)
                dates.append(d); places.append(p); weathers.append(w)
                intervals.append(calc_days_between(race_date, d))
            data_dict[f"{prefix}の日付"] = dates
            data_dict[f"{prefix}の競馬場"] = places
            data_dict[f"{prefix}の天候"] = weathers
            data_dict[f"{prefix}からの日数"] = intervals

            current_past_field_index = 1
            continue

        # --- 過去レースの詳細項目解析 ---
        if (current_past_num is not None and current_past_field_index is not None and current_past_field_index < len(past_race_fields)):
            field_name = past_race_fields[current_past_field_index]
            prefix = "前走" if current_past_num == 1 else f"{current_past_num}走前"

            if field_name == "成績":
                dates, places, weathers, intervals = [], [], [], []
                for val in row_values:
                    d, p, w = split_date_place_weather(val)
                    dates.append(d); places.append(p); weathers.append(w)
                    intervals.append(calc_days_between(race_date, d))
                data_dict[f"{prefix}の日付"] = dates
                data_dict[f"{prefix}の競馬場"] = places
                data_dict[f"{prefix}の天候"] = weathers
                data_dict[f"{prefix}からの日数"] = intervals

            elif field_name == "コース":
                tracks, dists, conds = [], [], []
                for val in row_values:
                    t, d, c = split_course_info(val)
                    tracks.append(t); dists.append(d); conds.append(c)
                data_dict[f"{prefix}の芝orダート"] = tracks
                data_dict[f"{prefix}の距離"] = dists
                data_dict[f"{prefix}の馬場"] = conds

            elif field_name == "騎手,斤量":
                jockeys, weights = [], []
                for val in row_values:
                    j, w = split_jockey_weight(val)
                    jockeys.append(j); weights.append(w)
                data_dict[f"{prefix}の騎手"] = jockeys
                data_dict[f"{prefix}の斤量"] = weights

            elif field_name == "頭数,馬番,人気":
                heads, nums, pops = [], [], []
                for val in row_values:
                    h, n, p = split_head_num_pop(val)
                    heads.append(h); nums.append(n); pops.append(p)
                data_dict[f"{prefix}の頭数"] = heads
                data_dict[f"{prefix}の馬番"] = nums
                data_dict[f"{prefix}の人気"] = pops

            elif field_name == "タイム,(着順)":
                times, finishes = [], []
                for val in row_values:
                    rt, f = split_time_and_finish(val)
                    times.append(rt); finishes.append(f)
                data_dict[f"{prefix}のタイム"] = times
                data_dict[f"{prefix}の着順"] = finishes

            elif field_name == "ペース,脚質,上3F":
                paces, legs, times = [], [], []
                for val in row_values:
                    p, l, t = split_pace_leg_3f(val)
                    paces.append(p); legs.append(l); times.append(t)
                data_dict[f"{prefix}のペース"] = paces
                data_dict[f"{prefix}の脚質"] = legs
                data_dict[f"{prefix}の上3F"] = times

            elif field_name == "トップ(タイム差)":
                tops, margins = [], []
                for val in row_values:
                    t, m = split_top_margin(val)
                    tops.append(t); margins.append(m)
                data_dict[f"{prefix}のトップ馬"] = tops
                data_dict[f"{prefix}のタイム差"] = margins

            elif field_name == "馬体重()3F順":
                bw, bc, r3 = [], [], []
                for val in row_values:
                    w, c, r = split_weight_change_3f(val)
                    bw.append(w); bc.append(c); r3.append(r)
                data_dict[f"{prefix}の馬体重"] = bw
                data_dict[f"{prefix}の体重増減"] = bc
                data_dict[f"{prefix}の上がり3F順位"] = r3

            else:
                data_dict[f"{prefix}の{field_name}"] = row_values

            current_past_field_index += 1
            if current_past_field_index >= len(past_race_fields):
                current_past_num = None
                current_past_field_index = None
            continue

        # --- 通常の項目 ---
        if row_title:
            data_dict[row_title] = row_values
            if row_title == "調教師":
                current_race_field_index = 0
            continue

    if "馬番" not in data_dict:
        return None

    year = 2000 + int(url_code[0:2])
    place = int(url_code[2:4])
    kai = int(url_code[4:6])
    day = int(url_code[6:8])
    race_num = int(url_code[8:10])

    horses = []
    for i in range(num_horses):
        horse = {
            "年": year, "競馬場": place, "回": kai, "日": day,
            "レース": race_num, "URLコード": url_code,
            "芝orダート": track_type, "距離": distance,
            "回り": direction, "馬場": track_condition,
        }
        for key, values in data_dict.items():
            val = values[i] if i < len(values) else ""
            if key == "性齢":
                val_str = str(val).strip()
                # 「せん」「セン」「セ」「騙」「牡」「牝」に対応する正規表現に修正
                match = re.match(r"^(牡|牝|セ|セン|せん|騙)(\d+)", val_str)
                if match:
                    horse["性別"] = match.group(1)
                    horse["年齢"] = match.group(2)
                else:
                    horse["性別"] = val_str
                    horse["年齢"] = ""
            else:
                horse[key] = val
        horses.append(horse)

    horses.sort(key=lambda x: int(re.search(r"\d+", str(x.get("馬番", "999"))).group()) if re.search(r"\d+", str(x.get("馬番", ""))) else 999)
    return horses

# ============================================================
# メイン処理
# ============================================================

def main():
    required_columns = [
        "芝orダート", "距離", "回り", "馬場",
        "馬体重", "上がり3F順位", "ペース", "脚質", 
        "前走の日付", "前走の天候", "前走の騎手", 
        "前走の頭数", "前走の人気", "前走のタイム差",
        "前走からの日数",
        "父馬", "母馬", "母父馬",
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    csv_dir = os.path.join(base_dir, "CSV_predict")
    os.makedirs(csv_dir, exist_ok=True)
    print(f"保存先フォルダ: {csv_dir}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    })

    for y in TARGET_YEARS:
        print(f"\n=== {2000 + y}年の処理を開始します ===")
        for p in PLACES:
            for k in KAIS:
                for d in DAYS:
                    day_exists = False
                    for r in RACES:
                        code = f"{y:02d}{p:02d}{k:02d}{d:02d}{r:02d}"
                        csv_file = os.path.join(csv_dir, f"{CSV_PREFIX}_{code}.csv")
                        needs_scraping = True

                        if os.path.exists(csv_file):
                            try:
                                if RESCRAPE_IF_COLUMNS_MISSING:
                                    df = pd.read_csv(csv_file, nrows=0)
                                    missing = [col for col in required_columns if col not in df.columns]
                                    if not missing:
                                        needs_scraping = False
                                        day_exists = True
                                        print(f"[既存・スキップ] {code}")
                                    else:
                                        print(f"[再取得] {code} (不足: {','.join(missing)})")
                                else:
                                    needs_scraping = False
                                    day_exists = True
                            except Exception as e:
                                print(f"[CSV読込エラー] {csv_file}: {e}")

                        if not needs_scraping:
                            continue

                        url = BASE_URL + code
                        time.sleep(REQUEST_INTERVAL_SECONDS)

                        try:
                            res = session.get(url, timeout=15, allow_redirects=True)
                            res.raise_for_status()

                            if f"code={code}" not in res.url:
                                break

                            res.encoding = "cp932"
                            match = re.search(r"dbcl2\(['\"]?(\d{10})['\"]?\)", res.text)
                            if match and match.group(1) != code:
                                break

                            horses = parse_race_html(res.text, code)
                            if horses:
                                pd.DataFrame(horses).to_csv(csv_file, index=False, encoding="utf-8-sig")
                                day_exists = True
                                print(f"[取得成功] {url} ({len(horses)}頭)")
                            else:
                                if r == 1:
                                    break
                        except Exception as e:
                            print(f"[通信/処理エラー] {url} : {e}")
                            time.sleep(5)

                    if not day_exists:
                        break

    print("\n=== 全処理が完了しました ===")

if __name__ == "__main__":
    main()