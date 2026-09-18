import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests
from bs4 import BeautifulSoup

import fetch_odds
import schedule_bets as scheduler
from suggest_bets import DISCORD_BET_RULES, Horse, meets_discord_rules, suggest_for_race


class DiscordFilterTests(unittest.TestCase):
    def test_filter_before_rounding(self):
        horses = [Horse({"馬番": 1, "馬名": "テスト", "1着確率": 1.0,
                         "2着以内確率": 1.0, "3着以内確率": 0.5})]
        for odds, accepted in [(2.3992, False), (2.4, True), (2.4008, True)]:
            with self.subTest(odds=odds):
                frames = {"tanpuku": pd.DataFrame([{"馬番": 1, "単勝オッズ": 1000001,
                                                  "複勝オッズ下限": odds}])}
                bets = suggest_for_race(horses, frames, 1.0, discord_only=True)
                self.assertEqual(bool(bets), accepted)
                self.assertTrue(all(b["式別"] == "複勝" for b in bets))


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.race = scheduler.Race("202606040601", "テストレース", datetime(2026, 9, 20, 10, tzinfo=scheduler.JST))
        self.now = lambda: self.race.start - timedelta(minutes=5)
        self.state_path = self.base / "suggestions" / "discord_state" / f"{self.race.race_id}.json"
        self.webhook = "https://discord.com/api/webhooks/123/secret"

    def predictions(self):
        folder = self.base / "predictions"
        folder.mkdir()
        pd.DataFrame([{"開催日": "2026年 第4回6日目", "競馬場": "中山", "レース番号": "1R",
                       "馬番": n, "馬名": f"馬{n}", "1着確率": p,
                       "2着以内確率": 0.7, "3着以内確率": 0.8}
                      for n, p in [(1, 0.5), (2, 0.3), (3, 0.2)]]).to_csv(
            folder / "pred_2026_第4回中山6日目_1R.csv", index=False, encoding="utf-8-sig")

    def test_all_rule_boundaries(self):
        for name, (prob, low, high) in DISCORD_BET_RULES.items():
            with self.subTest(name=name):
                self.assertTrue(meets_discord_rules(name, prob, low))
                self.assertFalse(meets_discord_rules(name, prob - 0.00001, low))
                self.assertFalse(meets_discord_rules(name, prob, low - 0.00001))
                if math.isfinite(high):
                    self.assertTrue(meets_discord_rules(name, prob, high))
                    self.assertFalse(meets_discord_rules(name, prob, high + 0.00001))
        for name in ["単勝", "3連単", "枠連"]:
            self.assertFalse(meets_discord_rules(name, 1, 2))
        self.assertFalse(meets_discord_rules("複勝", float("nan"), 2))

    def test_ids_and_dates(self):
        for value in ["2606040601", "202606040601"]:
            self.assertEqual(fetch_odds.normalize_race_id(value), self.race.race_id)
        html = '<div id="RaceList_DateList"><li class="Active"><a href="?kaisai_date=20260920">9月20日</a></li></div><div class="RaceData01">10:00発走</div><div class="RaceName">テストレース</div>'
        self.assertEqual(scheduler.parse_race_page("2606040601", html), self.race)
        with self.assertRaises(ValueError):
            scheduler.parse_race_page(self.race.race_id, '<div class="RaceData01">10:00発走</div>')
        for value in ["202611040601", "202606040613", "invalid"]:
            with self.assertRaises(ValueError):
                fetch_odds.normalize_race_id(value)

    def test_due_boundaries(self):
        self.assertFalse(scheduler.due(self.race, self.now() - timedelta(seconds=1)))
        self.assertTrue(scheduler.due(self.race, self.now()))
        self.assertTrue(scheduler.due(self.race, self.race.start - timedelta(seconds=1)))
        self.assertFalse(scheduler.due(self.race, self.race.start))

    def test_pipeline_and_restart(self):
        self.predictions()
        odds = {"tanpuku": pd.DataFrame([{"馬番": 1, "単勝オッズ": 50, "複勝オッズ下限": 2}]),
                "umaren": pd.DataFrame([{"馬番1": 1, "馬番2": 2, "馬連オッズ": 3}])}
        with patch.object(scheduler, "fetch_live_race_odds", return_value=odds) as fetch, patch.object(scheduler, "send_discord") as send:
            scheduler.process_race(self.race, self.webhook, self.base, self.now)
            fetch.assert_called_once_with(self.race.race_id, "20260920", self.base)
            message = send.call_args.args[1]
            self.assertIn("中山 1R", message)
            self.assertIn("複勝", message)
            self.assertIn("馬連", message)
            self.assertNotIn("単勝", message)
            scheduler.process_race(self.race, self.webhook, self.base, self.now)
            send.assert_called_once()
        self.assertTrue(json.loads(self.state_path.read_text(encoding="utf-8"))["done"])

    def test_no_bets(self):
        self.predictions()
        with patch.object(scheduler, "fetch_live_race_odds", return_value={}), patch.object(scheduler, "send_discord") as send:
            scheduler.process_race(self.race, self.webhook, self.base, self.now)
            self.assertIn("指定条件に合う買い目はありません", send.call_args.args[1])

    def test_failure_is_not_no_bets(self):
        with patch.object(scheduler, "send_discord") as send:
            scheduler.process_race(self.race, self.webhook, self.base, self.now)
            self.assertIn("取得できませんでした", send.call_args.args[1])
            self.assertNotIn("指定条件に合う買い目はありません", send.call_args.args[1])

    def test_resume_partial_messages(self):
        scheduler.save_state(self.state_path, {"messages": ["first", "second", "third"], "sent": 0, "done": False})
        with patch.object(scheduler, "send_discord", side_effect=[None, RuntimeError("failed")]):
            with self.assertRaises(RuntimeError):
                scheduler.process_race(self.race, self.webhook, self.base, self.now)
        with patch.object(scheduler, "send_discord") as send, patch.object(scheduler, "fetch_live_race_odds") as fetch:
            scheduler.process_race(self.race, self.webhook, self.base, self.now)
            self.assertEqual([call.args[1] for call in send.call_args_list], ["second", "third"])
            fetch.assert_not_called()

    def test_no_send_after_start(self):
        with patch.object(scheduler, "send_discord") as send, patch.object(scheduler, "fetch_live_race_odds") as fetch:
            scheduler.process_race(self.race, self.webhook, self.base, lambda: self.race.start)
            send.assert_not_called()
            fetch.assert_not_called()

    def test_split_messages(self):
        bets = [{"式別": "馬連", "買い目": f"1-{i}", "的中確率(推定)": 0.2, "オッズ": 10, "期待値": 2} for i in range(100)]
        messages = scheduler.build_messages(self.race, bets)
        self.assertGreater(len(messages), 1)
        self.assertTrue(all(len(message) <= 1900 and message.startswith(self.race.label) for message in messages))
        self.assertEqual(sum(message.count("| 確率") for message in messages), 100)

    def test_webhook_error_redacted(self):
        session = Mock()
        session.post.side_effect = requests.ConnectionError(self.webhook)
        with self.assertRaises(RuntimeError) as caught:
            scheduler.send_discord(self.webhook, "test", session)
        self.assertNotIn("secret", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        session.post.side_effect = None
        session.post.return_value.status_code = 200
        scheduler.send_discord(self.webhook, "test", session)
        self.assertEqual(session.post.call_args.kwargs["json"]["allowed_mentions"], {"parse": []})

    def test_live_fetch_only_target_and_missing_odds(self):
        def link(cname):
            return f'<a onclick="doAction(\'/JRADB/accessO.html\', \'{cname}\')">odds</a>'
        meeting = "pw15orl00062026040620260920/00"
        cnames = {key: f"pw{code}ouS306202604060120260920Z/00" for code, key in [("151", "tanpuku"), ("154", "umaren"), ("155", "wide"), ("156", "umatan"), ("157", "fuku3")]}
        top = BeautifulSoup(link(meeting), "html.parser")
        meeting_page = BeautifulSoup("".join(link(c) for c in cnames.values()) + link("pw151ouS306202604060220260920Z/00"), "html.parser")
        empty = BeautifulSoup("", "html.parser")
        client = Mock()
        client.fetch.side_effect = [top, meeting_page, empty, empty, empty, empty, empty]
        with patch.object(fetch_odds, "JraOddsClient", return_value=client), patch.object(fetch_odds, "parse_tanpuku", return_value={1: {"馬番": 1, "馬名": "馬1"}}), patch.object(fetch_odds, "parse_pair_tables", return_value=[(1, 2, "3.0")]), patch.object(fetch_odds, "parse_fuku3", return_value=[(1, 2, 3, "4.0")]):
            result = fetch_odds.fetch_live_race_odds(self.race.race_id, "20260920", self.base)
        self.assertEqual(set(result), set(cnames))
        self.assertEqual([c.args[0] for c in client.fetch.call_args_list], [fetch_odds.TOP_PAGE_CNAME, meeting, *cnames.values()])
        client.session.close.assert_called_once()
        client.reset_mock()
        client.fetch.side_effect = [top, BeautifulSoup(link(cnames["tanpuku"]), "html.parser"), empty]
        with patch.object(fetch_odds, "JraOddsClient", return_value=client), patch.object(fetch_odds, "save_odds_csvs") as save:
            with self.assertRaises(RuntimeError):
                fetch_odds.fetch_live_race_odds(self.race.race_id, "20260920", self.base)
            save.assert_not_called()
        client.session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
