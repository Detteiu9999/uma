"""discord_state を動的に集計するローカル購入成績ビューアー。"""

import argparse
import json
import re
import threading
import time
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests
from bs4 import BeautifulSoup

from ipat_auto_bet import parse_bet_combo
from schedule_bets import JST, NETKEIBA_RESULT_URL, parse_race_result

BASE_DIR = Path(__file__).resolve().parent
BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
HEADER = re.compile(r"^(.+?) (\d+)R (.*?) 発走 (\d{4}/\d{2}/\d{2} \d{2}:\d{2}) JST$", re.M)
BET_LINE = re.compile(r"^(単勝|複勝|枠連|馬連|ワイド|馬単|3連複|3連単) (.+?) \| 確率 .+? \| オッズ .+? \| 期待値 .+$", re.M)
PURCHASE = re.compile(r"IPATで (\d+)点 ([\d,]+)円を自動購入しました")


def read_races(state_dir):
    """旧形式は送信済みのメッセージから復元。購入は成功記録のみ採用する。"""
    races, warnings = [], []
    if not state_dir.is_dir():
        return [], [f"データフォルダがありません: {state_dir}"]
    for path in sorted(state_dir.glob("*.json")):
        if not re.fullmatch(r"\d{12}", path.stem):
            continue  # schedule_ / summary_ はレース・購入記録ではない
        try:
            state = json.loads(path.read_text(encoding="utf-8-sig"))
            messages = state.get("messages", [])
            if not isinstance(messages, list) or not all(isinstance(m, str) for m in messages):
                raise ValueError("messages の形式が不正です")
            header = HEADER.search("\n".join(messages))
            if not header:
                raise ValueError("発走日時を取得できません")
            start = datetime.strptime(header[4], "%Y/%m/%d %H:%M").replace(tzinfo=JST)
            sent = state.get("sent", len(messages) if state.get("done") else 0)
            if type(sent) is not int or not 0 <= sent <= len(messages):
                raise ValueError("送信件数が不正です")
            notified = [{"式別": m[1], "買い目": m[2]}
                        for m in BET_LINE.finditer("\n".join(messages[:sent]))]
            bets = state.get("bets", notified)
            if not isinstance(bets, list):
                raise ValueError("bets の形式が不正です")
            for bet in bets + notified:
                if not isinstance(bet, dict) or parse_bet_combo(bet.get("式別"), bet.get("買い目")) is None:
                    raise ValueError("解析できない買い目があります")
            receipts = {m.group(0) for text in [str(state.get("auto_bet_result", "")), *messages]
                        for m in PURCHASE.finditer(text)}
            amount = None
            if receipts:
                if len(receipts) != 1:
                    raise ValueError("購入成功記録が一致しません")
                receipt = PURCHASE.fullmatch(receipts.pop())
                points, total = int(receipt[1]), int(receipt[2].replace(",", ""))
                if not points or points != len(bets) or total <= 0 or total % (points * 100):
                    raise ValueError("購入点数・金額と買い目が一致しません")
                amount = total // points
            races.append({"id": path.stem, "date": start.date().isoformat(),
                          "start": start.isoformat(), "place": header[1],
                          "number": int(header[2]), "name": header[3],
                          "bets": bets, "notified": notified, "stake": amount})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            warnings.append(f"{path.name}: 読み込み対象外 ({exc})")
    return races, warnings


def validate_payouts(payouts):
    if not isinstance(payouts, dict) or not payouts:
        raise ValueError("払戻テーブルが空です")
    for kind, entries in payouts.items():
        if kind not in BET_TYPES or not isinstance(entries, list) or not entries:
            raise ValueError("払戻テーブルの形式が不正です")
        for combo, value in entries:
            if (not isinstance(combo, (tuple, list)) or not combo
                    or any(type(n) is not int for n in combo)
                    or type(value) is not int or value < 0):
                raise ValueError("払戻額・組み合わせの形式が不正です")
    return payouts


def fetch_payouts(race_id, session):
    response = session.get(NETKEIBA_RESULT_URL,
                           params={"race_id": race_id}, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    # 通常の払戻テーブルだけでは取消馬の返還や特払を正しく評価できない。
    text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
    if re.search(r"返還|特払|特別払戻", text):
        raise ValueError("返還・特払を含むため自動集計できません")
    return validate_payouts(parse_race_result(race_id, response.text))


def totals(tickets):
    settled = [t for t in tickets if t["return"] is not None]
    purchased = sum(t["bet"] for t in tickets)
    bet = sum(t["bet"] for t in settled)
    returned = sum(t["return"] for t in settled)
    return {"points": len(tickets), "settled_points": len(settled),
            "hits": sum(t["return"] > 0 for t in settled),
            "purchased": purchased, "bet": bet, "pending_bet": purchased - bet,
            "return": returned, "profit": returned - bet,
            "roi": returned / bet * 100 if bet else None}


class Dashboard:
    def __init__(self, base_dir=BASE_DIR, auto_fetch=True):
        self.state_dir = Path(base_dir) / "suggestions" / "discord_state"
        self.cache_dir = Path(base_dir) / "suggestions" / "dashboard_results"
        self.auto_fetch = auto_fetch
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.errors = {}
        self.retry_after = {}
        self.progress = {"running": False, "remaining": 0, "race_id": None}

    def load_payouts(self, race_id):
        path = self.cache_dir / f"{race_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["race_id"] != race_id:
                raise ValueError("レースIDが一致しません")
            return validate_payouts(data["payouts"])
        except (OSError, ValueError, KeyError, TypeError):
            return None  # 破損したキャッシュは再取得する

    def save_payouts(self, race_id, payouts):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{race_id}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"race_id": race_id, "fetched_at": datetime.now(JST).isoformat(),
                                    "payouts": payouts}, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)

    def sync_once(self, session, now=None):
        now = now or datetime.now(JST)
        races, _ = read_races(self.state_dir)
        pending = [r for r in races if (r["bets"] or r["notified"])
                   and datetime.fromisoformat(r["start"]) + timedelta(minutes=20) <= now
                   and self.load_payouts(r["id"]) is None
                   and time.monotonic() >= self.retry_after.get(r["id"], 0)]
        # 実購入・直近のレースを優先して取得する。
        pending.sort(key=lambda r: (r["stake"] is not None, r["start"]), reverse=True)
        try:
            for index, race in enumerate(pending):
                if self.stop.is_set():
                    break
                race_id = race["id"]
                with self.lock:
                    self.progress = {"running": True, "remaining": len(pending) - index, "race_id": race_id}
                try:
                    self.save_payouts(race_id, fetch_payouts(race_id, session))
                    with self.lock:
                        self.errors.pop(race_id, None)
                except (requests.RequestException, OSError, ValueError) as exc:
                    with self.lock:
                        self.errors[race_id] = (str(exc) if isinstance(exc, ValueError)
                                                else f"結果取得・保存失敗 ({type(exc).__name__})")
                    self.retry_after[race_id] = time.monotonic() + 300
                if self.stop.wait(1):
                    break
        finally:
            with self.lock:
                self.progress = {"running": False, "remaining": 0, "race_id": None}

    def worker(self):
        with requests.Session() as session:
            session.headers["User-Agent"] = "Mozilla/5.0"
            while not self.stop.is_set():
                self.sync_once(session)
                self.stop.wait(60)

    def snapshot(self, query):
        mode = query.get("mode", "actual")
        if mode not in {"actual", "simulation"}:
            raise ValueError("集計モードが不正です")
        stake = int(query.get("stake", "100"))
        if not 100 <= stake <= 1000000 or stake % 100:
            raise ValueError("仮定購入額は100〜1,000,000円の100円単位で指定してください")
        for key in ("from", "to"):
            if query.get(key):
                datetime.strptime(query[key], "%Y-%m-%d")
        if query.get("from") and query.get("to") and query["from"] > query["to"]:
            raise ValueError("開始日は終了日以前にしてください")
        races, warnings = read_races(self.state_dir)
        with self.lock:
            progress, errors = dict(self.progress), dict(self.errors)
        rows, all_tickets = [], []
        daily, by_type = defaultdict(list), defaultdict(list)
        for race in races:
            if ((query.get("from") and race["date"] < query["from"])
                    or (query.get("to") and race["date"] > query["to"])
                    or (query.get("place") and race["place"] != query["place"])):
                continue
            if mode == "actual" and race["stake"] is None:
                continue
            bets = race["bets"] if mode == "actual" else race["notified"]
            unit = race["stake"] if mode == "actual" else stake
            payouts = self.load_payouts(race["id"])
            tickets = []
            for bet in bets:
                kind = bet["式別"]
                if query.get("type") and kind != query["type"]:
                    continue
                returned = None
                if payouts and kind in payouts:
                    combo = parse_bet_combo(kind, bet["買い目"])
                    returned = next((value * unit // 100 for nums, value in payouts[kind]
                                     if tuple(nums) == combo), 0)
                ticket = {"type": kind, "combo": bet["買い目"], "bet": unit, "return": returned}
                tickets.append(ticket)
                by_type[kind].append(ticket)
            if not tickets:
                continue
            all_tickets.extend(tickets)
            daily[race["date"]].extend(tickets)
            summary = totals(tickets)
            reason = None
            if summary["pending_bet"]:
                reason = errors.get(race["id"], "払戻情報待ち" if payouts is None else "対象式別の払戻情報がありません")
            rows.append({key: race[key] for key in ("id", "date", "start", "place", "number", "name")} |
                        {"tickets": tickets, "summary": summary, "pending_reason": reason})
        days = []
        cumulative = 0
        for date, tickets in sorted(daily.items()):
            summary = totals(tickets)
            cumulative += summary["profit"]
            days.append({"date": date, **summary, "cumulative_profit": cumulative})
        return {"updated_at": datetime.now(JST).isoformat(), "mode": mode,
                "summary": totals(all_tickets), "daily": days,
                "by_type": [{"type": kind, **totals(by_type[kind])} for kind in BET_TYPES if kind in by_type],
                "races": sorted(rows, key=lambda r: (r["start"], r["id"]), reverse=True),
                "places": sorted({r["place"] for r in races}),
                "years": sorted({int(r["date"][:4]) for r in races}, reverse=True), "warnings": warnings,
                "source": {"races": len(races), "purchased_races": sum(r["stake"] is not None for r in races),
                           "unconfirmed_races": sum(bool(r["bets"] or r["notified"]) and r["stake"] is None for r in races)},
                "sync": {**progress, "enabled": self.auto_fetch}}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, dashboard, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, **kwargs)

    def respond(self, status, content, content_type):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)
        except ConnectionError:
            pass  # 条件の連続変更やタブを閉じた際の fetch 中断

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/api/stats":
            try:
                data = self.dashboard.snapshot({k: v[-1] for k, v in parse_qs(url.query).items()})
                status = 200
            except (ValueError, TypeError) as exc:
                status, data = 400, {"error": str(exc)}
            self.respond(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                         "application/json; charset=utf-8")
            return
        assets = {"/": ("index.html", "text/html"), "/dashboard.js": ("dashboard.js", "text/javascript"),
                  "/dashboard.css": ("dashboard.css", "text/css")}
        if url.path in assets:
            name, content_type = assets[url.path]
            self.respond(200, (BASE_DIR / "dashboard" / name).read_bytes(), f"{content_type}; charset=utf-8")
        else:
            self.respond(404, b"Not found", "text/plain")

    def log_message(self, format, *args):
        if len(args) < 2 or str(args[1]) != "200":
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Discord購入記録の成績をブラウザに表示します")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR, help="suggestions を含むプロジェクトフォルダ")
    parser.add_argument("--no-fetch", action="store_true", help="結果の自動取得を停止（保存済み結果のみ使用）")
    parser.add_argument("--open", action="store_true", help="ブラウザを自動で開く")
    args = parser.parse_args()
    dashboard = Dashboard(args.base_dir, auto_fetch=not args.no_fetch)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, dashboard=dashboard))
    if dashboard.auto_fetch:
        threading.Thread(target=dashboard.worker, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"購入成績: {url}\n終了するには Ctrl+C を押してください。", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
