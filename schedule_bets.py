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

from fetch_odds import PLACE_NAMES, fetch_live_race_odds, normalize_race_id
from suggest_bets import Horse, suggest_for_race

JST = timezone(timedelta(hours=9))
BASE_DIR = Path(__file__).resolve().parent
NETKEIBA_URL = "https://race.netkeiba.com/race/shutuba.html"
DISCORD_WEBHOOK_URL = (
    "https://discord.com/api/webhooks/1550427299026051172/LB_ZE6LasghBXObtBjJQ9JXOyMwnsL9L8aGRZfQdpOF8CPE-xk2YDFTXnBz9gK1RgS8w"
)

@dataclass(frozen=True)
class Race:
    race_id: str
    name: str
    start: datetime

    @property
    def label(self):
        return (f"{PLACE_NAMES[int(self.race_id[4:6])]} "
                f"{int(self.race_id[-2:])}R {self.name} "
                f"発走 {self.start:%Y/%m/%d %H:%M} JST")


def discover_race_ids(predict_dir):
    ids = set()
    for path in Path(predict_dir).glob("horse_racing_data_*.csv"):
        match = re.fullmatch(r"horse_racing_data_([0-9]{10}|[0-9]{12})\.csv", path.name)
        if match:
            ids.add(normalize_race_id(match[1]))
    return sorted(ids)


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
    return Race(race_id, title.get_text(" ", strip=True) if title else "", start)


def fetch_schedule(race_id, session):
    response = session.get(NETKEIBA_URL, params={"race_id": race_id, "rf": "race_list"}, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return parse_race_page(race_id, response.text)


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
    current = "\n" + header  # 先頭に空行を入れる

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


def process_race(race, webhook, base_dir=BASE_DIR, now=None):
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
            messages = build_messages(race, [], "予測CSV・最新オッズの取得または計算に失敗しました。ログを確認してください")
        state = {"messages": messages, "sent": 0, "done": False}
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
    return race.start - timedelta(minutes=5) <= now < race.start


def main():
    parser = argparse.ArgumentParser(description="CSV_predictのレースを発走5分前に処理してDiscord通知（JST）")
    parser.add_argument("--list", action="store_true", help="発走日時を取得・表示して終了（送信なし）")
    args = parser.parse_args()
    webhook = None
    if not args.list:
        try:
            webhook = validate_webhook(DISCORD_WEBHOOK_URL)
        except ValueError as exc:
            parser.error(
                "コード内の DISCORD_WEBHOOK_URL をDiscordのWebhook URLに置き換えてください: "
                + str(exc)
            )
    races = []
    failed = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "Mozilla/5.0"
        for race_id in discover_race_ids(BASE_DIR / "CSV_predict"):
            try:
                race = fetch_schedule(race_id, session)
                print(f"{race.label} / 実行 {race.start - timedelta(minutes=5):%H:%M}", flush=True)
                races.append(race)
            except Exception as exc:
                print(f"[{race_id}] 発走日時取得失敗 ({type(exc).__name__})", flush=True)
                failed.append(race_id)
            time.sleep(1)
    if failed:
        raise SystemExit("発走日時を取得できないレースがあります。確認後、再実行してください。")
    if args.list:
        return
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
                        print(f"[{race_id}] 通知処理失敗 ({type(exc).__name__})。30秒後に再試行", flush=True)
                        retry_at[race_id] = now + timedelta(seconds=30)
            for race_id, race in list(pending.items()):
                if race_id in running:
                    continue
                if now >= race.start:
                    print(f"[{race_id}] 発走済みのためスキップ", flush=True)
                    del pending[race_id]
                elif due(race, now) and now >= retry_at.get(race_id, now):
                    running[race_id] = executor.submit(process_race, race, webhook)
            if pending or running:
                time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("監視を終了しました。")
