import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from discord_webhook_url import DISCORD_SCHEDULE_WEBHOOK_URL, DISCORD_WEBHOOK_URL
from fetch_odds import PLACE_NAMES, fetch_live_race_odds, normalize_race_id
from ipat_auto_bet import IpatError, IpatSession, buy_race_bets
from suggest_bets import Horse, suggest_for_race

JST = timezone(timedelta(hours=9))
BASE_DIR = Path(__file__).resolve().parent
NETKEIBA_URL = "https://race.netkeiba.com/race/shutuba.html"
NETKEIBA_RESULT_URL = "https://race.netkeiba.com/race/result.html"
RESULT_WAIT_MINUTES = 60  # 最終レース発走後に結果取得を待つ上限時間
AUTO_BET_YEN = 300  # 自動購入の1点あたり金額
NOTIFY_BEFORE_MINUTES = 3  # 買い目通知・自動購入を行うタイミング（発走N分前）

@dataclass(frozen=True)
class Race:
    race_id: str
    name: str
    start: datetime
    is_obstacle: bool = False

    @property
    def short_label(self):
        return f"{PLACE_NAMES[int(self.race_id[4:6])]} {int(self.race_id[-2:])}R"

    @property
    def label(self):
        return f"{self.short_label} {self.name} 発走 {self.start:%Y/%m/%d %H:%M} JST"


def discover_race_ids(predict_dir):
    ids = set()
    for path in Path(predict_dir).glob("horse_racing_data_*.csv"):
        match = re.fullmatch(r"horse_racing_data_([0-9]{10}|[0-9]{12})\.csv", path.name)
        if match:
            ids.add(normalize_race_id(match[1]))
    return sorted(ids)


def is_obstacle_race(html):
    """出馬表ページのレース条件に「障」（障害レース）が含まれるか判定する。
    netkeiba では障害レースが「障3000m」のように表記される。"""
    soup = BeautifulSoup(html, "html.parser")
    data = soup.select_one(".RaceData01")
    return bool(data and re.search(r"障(?:害)?[0-9]", data.get_text(" ", strip=True)))


def parse_race_page(race_id, html):
    race_id = normalize_race_id(race_id)
    soup = BeautifulSoup(html, "html.parser")
    data = soup.select_one(".RaceData01")
    match = re.search(r"([0-2]?[0-9]):([0-5][0-9])\s*発走", data.get_text(" ", strip=True) if data else "")
    if not match:
        raise ValueError("発走時刻を取得できません（未公開・中止の可能性）")
    dates = set()
    active = soup.select("#RaceList_DateList .Active, .RaceList_DateList .Active")
    for node in active:
        for link in [node, *node.select("a[href]")]:
            value = parse_qs(urlparse(link.get("href", "")).query).get("kaisai_date", [])
            dates.update(v for v in value if re.fullmatch(r"[0-9]{8}", v))
        text = node.get_text(" ", strip=True)
        day = re.search(r"(\d{1,2})月(\d{1,2})日", text)
        if day:
            dates.add(f"{race_id[:4]}{int(day[1]):02d}{int(day[2]):02d}")
    if len(dates) != 1:
        raise ValueError("開催日を一意に取得できません")
    date_str = dates.pop()
    if date_str[:4] != race_id[:4]:
        raise ValueError("開催年とレースIDが一致しません")
    start = datetime.strptime(f"{date_str} {match[1]}:{match[2]}", "%Y%m%d %H:%M").replace(tzinfo=JST)
    title = soup.select_one(".RaceName")
    obstacle = bool(data and re.search(r"障(?:害)?[0-9]", data.get_text(" ", strip=True)))
    return Race(race_id, title.get_text(" ", strip=True) if title else "", start, obstacle)


def fetch_schedule(race_id, session):
    response = session.get(NETKEIBA_URL, params={"race_id": race_id, "rf": "race_list"}, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return parse_race_page(race_id, response.text)


# 払戻テーブルの行クラス → 式別
PAYOUT_ROW_CLASSES = {
    "Tansho": "単勝", "Fukusho": "複勝", "Wakuren": "枠連", "Umaren": "馬連",
    "Wide": "ワイド", "Umatan": "馬単", "Fuku3": "3連複", "Tan3": "3連単",
}
UNORDERED_BETS = {"枠連", "馬連", "ワイド", "3連複"}


def parse_race_result(race_id, html):
    """結果ページの払戻テーブルを解析する。
    戻り値: {式別: [(買い目タプル, 100円あたり払戻額), ...]}（順不同の式別は昇順に正規化）
    結果未確定（払戻テーブルなし）の場合は ValueError。"""
    race_id = normalize_race_id(race_id)
    soup = BeautifulSoup(html, "html.parser")
    payouts = {}
    for row in soup.select("table.Payout_Detail_Table tr"):
        classes = row.get("class") or []
        bet_type = next((name for cls, name in PAYOUT_ROW_CLASSES.items() if cls in classes), None)
        result_td = row.find("td", class_="Result")
        payout_td = row.find("td", class_="Payout")
        if not bet_type or not result_td or not payout_td:
            continue
        combos = []
        if bet_type in ("単勝", "複勝"):
            combos = [(int(div.get_text(strip=True)),) for div in result_td.find_all("div")
                      if div.get_text(strip=True).isdigit()]
        else:
            for ul in result_td.find_all("ul"):
                nums = tuple(int(li.get_text(strip=True)) for li in ul.find_all("li")
                             if li.get_text(strip=True).isdigit())
                if nums:
                    combos.append(tuple(sorted(nums)) if bet_type in UNORDERED_BETS else nums)
        values = [int(m.group(1).replace(",", ""))
                  for text in payout_td.stripped_strings
                  if (m := re.fullmatch(r"([0-9][0-9,]*)円?", text))]
        entries = list(zip(combos, values))
        if entries:
            payouts.setdefault(bet_type, []).extend(entries)
    if not payouts:
        raise ValueError("払戻情報を取得できません（結果未確定の可能性）")
    return payouts


def fetch_race_result(race_id, session):
    response = session.get(NETKEIBA_RESULT_URL, params={"race_id": race_id, "rf": "race_list"}, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return parse_race_result(race_id, response.text)


def parse_suggestion_combo(bet_type, combo):
    """買い目文字列を照合用のタプルに変換する（順不同の式別は昇順に正規化）。"""
    match = None
    if bet_type in ("単勝", "複勝"):
        match = re.match(r"(\d+)", str(combo).strip())
    elif bet_type == "枠連":
        match = re.fullmatch(r"枠(\d+)-枠(\d+)", str(combo).strip())
    elif bet_type in ("馬連", "ワイド", "3連複"):
        match = re.fullmatch(r"(\d+)-(\d+)(?:-(\d+))?", str(combo).strip())
    elif bet_type in ("馬単", "3連単"):
        match = re.fullmatch(r"(\d+)→(\d+)(?:→(\d+))?", str(combo).strip())
    if not match:
        return None
    nums = tuple(int(n) for n in match.groups() if n is not None)
    return tuple(sorted(nums)) if bet_type in UNORDERED_BETS else nums


def evaluate_bets(bets, payouts, stake=100):
    """1レース分の買い目を確定払戻と照合し、購入点数・的中・払戻を集計する。"""
    hits = []
    for bet in bets:
        combo = parse_suggestion_combo(bet.get("式別"), bet.get("買い目"))
        if combo is None:
            continue
        for win_combo, value in payouts.get(bet["式別"], []):
            if win_combo == combo:
                hits.append({"式別": bet["式別"], "買い目": bet["買い目"],
                             "払戻": value * stake // 100})
                break
    return {"points": len(bets), "bet": stake * len(bets), "hits": hits,
            "hit_count": len(hits), "return": sum(hit["払戻"] for hit in hits)}


def load_race_horses(race_id, base_dir):
    year, place, kai, day, number = (int(race_id[:4]), int(race_id[4:6]),
                                   int(race_id[6:8]), int(race_id[8:10]), int(race_id[10:]))
    name = f"pred_{year}_第{kai}回{PLACE_NAMES[place]}{day}日目_{number}R.csv"
    path = Path(base_dir) / "predictions" / name
    if not path.exists():
        raise ValueError("予測CSVがありません。先に predict_model.py --predict-only を実行してください")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"開催日", "競馬場", "レース番号", "馬番", "馬名", "1着確率", "2着以内確率", "3着以内確率"}
    if df.empty or not required <= set(df.columns):
        raise ValueError("予測CSVの列が不足しているか空です")
    expected = {"開催日": f"{year}年 第{kai}回{day}日目", "競馬場": PLACE_NAMES[place], "レース番号": f"{number}R"}
    if any(not df[col].eq(value).all() for col, value in expected.items()):
        raise ValueError("予測CSVのレース識別情報が一致しません")
    return [Horse(row) for _, row in df.iterrows()]


def build_messages(race, suggestions, error=None):
    header = race.label
    if error:
        lines = [f"買い目を取得できませんでした: {error}"]
    else:
        lines = [f"{s['式別']} {s['買い目']} | 確率 {s['的中確率(推定)']:.2%} | "
                 f"オッズ {s['オッズ']:g} | 期待値 {s['期待値']:.3f}"
                 for s in sorted(suggestions, key=lambda s: -s["期待値"])]
        if not lines:
            lines = ["指定条件に合う買い目はありません。"]
        else:
            lines.append("\n複勝・ワイドはオッズ下限値で評価。")

    messages = []
    current = "_ _\n" + header  # 先頭に空行を入れる

    for line in lines:
        if len(current) + len(line) + 1 > 1900:
            messages.append(current)
            current = "\n" + header  # 分割後のメッセージにも空行を入れる
        current += "\n" + line

    messages.append(current)
    return messages


def validate_webhook(url):
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname not in {"discord.com", "discordapp.com"}
            or not re.fullmatch(r"/api/webhooks/[0-9]+/[^/]+", parsed.path)
            or parsed.username or parsed.password):
        raise ValueError("DISCORD_WEBHOOK_URL にDiscordのWebhook URLを設定してください")
    return url


def send_discord(url, content, session):
    for attempt in range(3):
        try:
            response = session.post(url, params={"wait": "true"},
                                    json={"content": content, "allowed_mentions": {"parse": []}}, timeout=20)
        except requests.RequestException:
            raise RuntimeError("Discordへの通信に失敗しました") from None
        if response.status_code == 429 and attempt < 2:
            try:
                delay = float(response.json().get("retry_after", 1))
            except (ValueError, TypeError):
                delay = 1
            time.sleep(max(1, min(delay, 60)))
            continue
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Discord送信失敗 (HTTP {response.status_code})")
        return
    raise RuntimeError("Discordのレート制限で送信できませんでした")


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def build_schedule_messages(races):
    """発走時刻順に並べた本日のレース一覧を、等幅のコードブロック表にして作る。"""
    first_start = min(race.start for race in races)
    header = f"本日のスケジュール ({first_start:%Y/%m/%d})"
    lines = [f"{race.start:%H:%M} {PLACE_NAMES[int(race.race_id[4:6])]} "
             f"{int(race.race_id[-2:]):>2}R {race.name}"
             for race in sorted(races, key=lambda r: (r.start, r.race_id))]
    messages = []
    current = "_ _\n" + header + "\n```"  # 先頭に空行を入れる
    for line in lines:
        # 閉じフェンス("\n```")の分を残して1900文字以内に収める
        if len(current) + len(line) + 5 > 1900:
            messages.append(current + "\n```")
            current = "\n" + header + "\n```"  # 分割後のメッセージにも空行を入れる
        current += "\n" + line
    messages.append(current + "\n```")
    return messages


def post_schedule(races, webhook, base_dir=BASE_DIR):
    """本日のレーススケジュール（発走時刻順）をスケジュール用チャンネルに投稿する。"""
    if not races:
        return
    first_start = min(race.start for race in races)
    state_path = Path(base_dir) / "suggestions" / "discord_state" / f"schedule_{first_start:%Y%m%d}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("done"):
        return
    with requests.Session() as session:
        for message in build_schedule_messages(races):
            send_discord(webhook, message, session)
    state["done"] = True
    save_state(state_path, state)
    print("本日のスケジュールをDiscordに投稿しました", flush=True)


def build_summary_messages(date, race_by_id, per_race, missing):
    total_points = sum(result["points"] for _, result in per_race)
    total_bet = sum(result["bet"] for _, result in per_race)
    total_return = sum(result["return"] for _, result in per_race)
    total_hits = sum(result["hit_count"] for _, result in per_race)
    lines = []
    if total_points:
        lines.append("全買い目を100円ずつ購入したと仮定")
        lines.append(f"購入: {total_points}点 {total_bet:,}円 / 的中: {total_hits}点")
        lines.append(f"払戻: {total_return:,}円 / 収支: {total_return - total_bet:+,}円")
        lines.append(f"回収率: {total_return / total_bet * 100:.1f}%")
        hits = [(race_id, hit) for race_id, result in per_race for hit in result["hits"]]
        if hits:
            lines.append("\n的中した買い目:")
            for race_id, hit in hits:
                lines.append(f"{race_by_id[race_id].short_label} {hit['式別']} "
                             f"{hit['買い目']} → {hit['払戻']:,}円")
    else:
        lines.append("本日の買い目はありませんでした。")
    if missing:
        labels = "、".join(race_by_id[race_id].short_label for race_id in missing)
        lines.append(f"\n※結果を取得できなかったため集計対象外: {labels}")
    header = f"本日の回収率 ({date:%Y/%m/%d})"
    messages = []
    current = "_ _\n" + header  # 先頭に空行を入れる
    for line in lines:
        if len(current) + len(line) + 1 > 1900:
            messages.append(current)
            current = "\n" + header  # 分割後のメッセージにも空行を入れる
        current += "\n" + line
    messages.append(current)
    return messages


def summarize_day(races, webhook, base_dir=BASE_DIR, now=None, sleep=time.sleep):
    """全レース終了後に、投稿した全買い目を100円ずつ購入したと仮定した
    本日の収支・回収率をDiscordに投稿する。"""
    if not races:
        return
    now = now or (lambda: datetime.now(JST))
    last_start = max(race.start for race in races)
    state_path = Path(base_dir) / "suggestions" / "discord_state" / f"summary_{last_start:%Y%m%d}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("done"):
        return
    race_by_id = {race.race_id: race for race in races}
    bets_by_race = {}
    for race in races:
        race_state_path = state_path.parent / f"{race.race_id}.json"
        if race_state_path.exists():
            race_state = json.loads(race_state_path.read_text(encoding="utf-8"))
            if race_state.get("bets"):
                bets_by_race[race.race_id] = race_state["bets"]
    results = {}
    deadline = last_start + timedelta(minutes=RESULT_WAIT_MINUTES)
    with requests.Session() as session:
        session.headers["User-Agent"] = "Mozilla/5.0"
        pending = set(bets_by_race)
        while pending:
            for race_id in sorted(pending):
                try:
                    results[race_id] = fetch_race_result(race_id, session)
                    pending.discard(race_id)
                    print(f"[{race_id}] 結果を取得しました", flush=True)
                except Exception as exc:
                    print(f"[{race_id}] 結果はまだ取得できません ({type(exc).__name__})", flush=True)
                sleep(1)
            if pending and now() < deadline:
                print(f"結果待ち: {len(pending)}レース。1分後に再取得します", flush=True)
                sleep(60)
            else:
                break
        if pending:
            print(f"結果を取得できなかったレース: {', '.join(sorted(pending))}", flush=True)
        per_race = [(race_id, evaluate_bets(bets, results[race_id]))
                    for race_id, bets in sorted(bets_by_race.items()) if race_id in results]
        messages = build_summary_messages(last_start, race_by_id, per_race, sorted(pending))
        for message in messages:
            send_discord(webhook, message, session)
    state["done"] = True
    save_state(state_path, state)
    print("本日の回収率をDiscordに投稿しました", flush=True)


def auto_bet_race(race, state, state_path, yen=AUTO_BET_YEN, session=None):
    """提案済みの買い目をIPATで自動購入する。障害レースは購入しない。
    結果メッセージ（Discord通知用）を返す。未実施なら None。"""
    if state.get("auto_bet_done"):
        return None
    if race.is_obstacle:
        state["auto_bet_done"] = True
        state["auto_bet_result"] = "障害レースのため自動購入しません"
        save_state(state_path, state)
        print(f"[{race.race_id}] 障害レースのため自動購入スキップ", flush=True)
        return None
    bets = state.get("bets") or []
    if not bets:
        state["auto_bet_done"] = True
        state["auto_bet_result"] = "買い目なしのため自動購入なし"
        save_state(state_path, state)
        return None
    try:
        points = buy_race_bets(race.race_id, bets, yen=yen, session=session)
    except (IpatError, requests.RequestException, ValueError) as exc:
        # 購入失敗は状態を保存せず、呼び出し側の再試行に委ねる
        raise RuntimeError(f"自動購入失敗: {exc}") from None
    state["auto_bet_done"] = True
    state["auto_bet_result"] = f"IPATで {points}点 {points * yen:,}円を自動購入しました"
    save_state(state_path, state)
    print(f"[{race.race_id}] {state['auto_bet_result']}", flush=True)
    return state["auto_bet_result"]


def process_race(race, webhook, base_dir=BASE_DIR, now=None, auto_bet=False, ipat_session=None):
    now = now or (lambda: datetime.now(JST))
    state_path = Path(base_dir) / "suggestions" / "discord_state" / f"{race.race_id}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("done") or now() >= race.start:
        return
    if "messages" not in state:
        try:
            horses = load_race_horses(race.race_id, base_dir)
            odds = fetch_live_race_odds(race.race_id, race.start.strftime("%Y%m%d"), base_dir)
            bets = suggest_for_race(horses, odds, 1.0, discord_only=True)
            messages = build_messages(race, bets)
        except Exception as exc:
            print(f"[{race.race_id}] 取得・計算失敗 ({type(exc).__name__})")
            bets = []
            messages = build_messages(race, [], "予測CSV・最新オッズの取得または計算に失敗しました。ログを確認してください")
        state = {"messages": messages, "bets": bets, "sent": 0, "done": False}
        save_state(state_path, state)
    if auto_bet and not state.get("auto_bet_done"):
        result = auto_bet_race(race, state, state_path, session=ipat_session)
        if result:
            state["messages"] = state["messages"] + [result]
            save_state(state_path, state)
    with requests.Session() as session:
        while state["sent"] < len(state["messages"]):
            if now() >= race.start:
                return
            send_discord(webhook, state["messages"][state["sent"]], session)
            state["sent"] += 1
            save_state(state_path, state)
    state["done"] = True
    save_state(state_path, state)


def due(race, now):
    return race.start - timedelta(minutes=NOTIFY_BEFORE_MINUTES) <= now < race.start


def main():
    parser = argparse.ArgumentParser(description=f"CSV_predictのレーススケジュールを別チャンネルに投稿し、発走{NOTIFY_BEFORE_MINUTES}分前に買い目をDiscord通知、全レース終了後に本日の回収率を投稿（JST）")
    parser.add_argument("--list", action="store_true", help="発走日時を取得・表示して終了（送信なし）")
    parser.add_argument("--auto-bet", action="store_true",
                        help=f"提案した買い目をIPATで{AUTO_BET_YEN}円ずつ自動購入する（障害レースは除く）")
    args = parser.parse_args()
    webhook = None
    schedule_webhook = None
    if not args.list:
        try:
            webhook = validate_webhook(DISCORD_WEBHOOK_URL)
        except ValueError as exc:
            parser.error(
                "コード内の DISCORD_WEBHOOK_URL をDiscordのWebhook URLに置き換えてください: "
                + str(exc)
            )
        if DISCORD_SCHEDULE_WEBHOOK_URL:
            try:
                schedule_webhook = validate_webhook(DISCORD_SCHEDULE_WEBHOOK_URL)
            except ValueError as exc:
                parser.error(
                    "コード内の DISCORD_SCHEDULE_WEBHOOK_URL をDiscordのWebhook URLに置き換えてください: "
                    + str(exc)
                )
    races = []
    failed = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "Mozilla/5.0"
        for race_id in discover_race_ids(BASE_DIR / "CSV_predict"):
            try:
                race = fetch_schedule(race_id, session)
                note = "（障害レース: 自動購入対象外）" if race.is_obstacle else ""
                print(f"{race.label}{note} / 実行 {race.start - timedelta(minutes=NOTIFY_BEFORE_MINUTES):%H:%M}", flush=True)
                races.append(race)
            except Exception as exc:
                print(f"[{race_id}] 発走日時取得失敗 ({type(exc).__name__})", flush=True)
                failed.append(race_id)
            time.sleep(1)
    if failed:
        raise SystemExit("発走日時を取得できないレースがあります。確認後、再実行してください。")
    if args.list:
        return
    if schedule_webhook:
        post_schedule(races, schedule_webhook)
    else:
        print("DISCORD_SCHEDULE_WEBHOOK_URL 未設定のためスケジュール投稿をスキップします", flush=True)
    if args.auto_bet and any(race.is_obstacle for race in races):
        print("障害レースは自動購入の対象外です", flush=True)
    ipat_session = None
    if args.auto_bet:
        # 1日1回だけログインし、全レースの購入でセッションを共有する
        # （セッション切れ時は buy_race_bets が再ログインして1回だけ再試行する）
        ipat_session = IpatSession()
        try:
            ipat_session.login()
        except Exception as exc:
            raise SystemExit(
                f"IPATログインに失敗しました ({type(exc).__name__}: {exc})。"
                "ipat_login_info.py を確認してください。")
        print("IPATにログインしました（自動購入モード）", flush=True)
    pending = {race.race_id: race for race in races if race.start > datetime.now(JST)}
    running = {}
    retry_at = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        while pending or running:
            now = datetime.now(JST)
            for race_id, future in list(running.items()):
                if future.done():
                    del running[race_id]
                    try:
                        future.result()
                        pending.pop(race_id, None)
                    except Exception as exc:
                        print(f"[{race_id}] 通知処理失敗 ({type(exc).__name__}: {exc})。30秒後に再試行", flush=True)
                        retry_at[race_id] = now + timedelta(seconds=30)
            for race_id, race in list(pending.items()):
                if race_id in running:
                    continue
                if now >= race.start:
                    print(f"[{race_id}] 発走済みのためスキップ", flush=True)
                    del pending[race_id]
                elif due(race, now) and now >= retry_at.get(race_id, now):
                    running[race_id] = executor.submit(
                        process_race, race, webhook, BASE_DIR, None, args.auto_bet, ipat_session)
            if pending or running:
                time.sleep(1)
    if ipat_session is not None:
        ipat_session.close()
    summarize_day(races, webhook)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("監視を終了しました。")
