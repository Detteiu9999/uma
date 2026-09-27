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
                         "2着以内確率": 0.5, "3着以内確率": 0.5})]
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


class ResultSummaryTests(unittest.TestCase):
    PAYOUT_HTML = """
    <table class="Payout_Detail_Table"><tbody>
    <tr class="Tansho"><th>単勝</th><td class="Result"><div><span>2</span></div><div><span></span></div><div><span></span></div></td><td class="Payout"><span>530円</span></td></tr>
    <tr class="Fukusho"><th>複勝</th><td class="Result"><div><span>2</span></div><div><span></span></div><div><span></span></div><div><span>1</span></div><div><span></span></div><div><span></span></div><div><span>14</span></div><div><span></span></div><div><span></span></div></td><td class="Payout"><span>160円<br />520円<br />140円</span></td></tr>
    <tr class="Wakuren"><th>枠連</th><td class="Result"><ul><li><span>1</span></li><li><span>2</span></li><li></li></ul></td><td class="Payout"><span>7,440円</span></td></tr>
    <tr class="Umaren"><th>馬連</th><td class="Result"><ul><li><span>1</span></li><li><span>2</span></li><li></li></ul></td><td class="Payout"><span>7,210円</span></td></tr>
    <tr class="Wide"><th>ワイド</th><td class="Result"><ul><li><span>1</span></li><li><span>2</span></li><li></li></ul><ul><li><span>2</span></li><li><span>14</span></li><li></li></ul><ul><li><span>1</span></li><li><span>14</span></li><li></li></ul></td><td class="Payout"><span>1,680円<br />400円<br />1,500円</span></td></tr>
    <tr class="Umatan"><th>馬単</th><td class="Result"><ul><li><span>2</span></li><li><span>1</span></li><li></li></ul></td><td class="Payout"><span>14,150円</span></td></tr>
    <tr class="Fuku3"><th>3連複</th><td class="Result"><ul><li><span>1</span></li><li><span>2</span></li><li><span>14</span></li></ul></td><td class="Payout"><span>5,840円</span></td></tr>
    <tr class="Tan3"><th>3連単</th><td class="Result"><ul><li><span>2</span></li><li><span>1</span></li><li><span>14</span></li></ul></td><td class="Payout"><span>59,180円</span></td></tr>
    </tbody></table>
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.race1 = scheduler.Race("202606040601", "レース1", datetime(2026, 9, 20, 10, tzinfo=scheduler.JST))
        self.race2 = scheduler.Race("202606040612", "レース2", datetime(2026, 9, 20, 16, tzinfo=scheduler.JST))
        self.webhook = "https://discord.com/api/webhooks/123/secret"
        self.summary_path = self.base / "suggestions" / "discord_state" / "summary_20260920.json"

    def save_bets(self, race, bets):
        scheduler.save_state(self.base / "suggestions" / "discord_state" / f"{race.race_id}.json",
                             {"messages": ["sent"], "bets": bets, "sent": 1, "done": True})

    def test_parse_race_result(self):
        payouts = scheduler.parse_race_result("202606040601", self.PAYOUT_HTML)
        self.assertEqual(payouts["単勝"], [((2,), 530)])
        self.assertEqual(payouts["複勝"], [((2,), 160), ((1,), 520), ((14,), 140)])
        self.assertEqual(payouts["枠連"], [((1, 2), 7440)])
        self.assertEqual(payouts["馬連"], [((1, 2), 7210)])
        self.assertEqual(payouts["ワイド"], [((1, 2), 1680), ((2, 14), 400), ((1, 14), 1500)])
        self.assertEqual(payouts["馬単"], [((2, 1), 14150)])
        self.assertEqual(payouts["3連複"], [((1, 2, 14), 5840)])
        self.assertEqual(payouts["3連単"], [((2, 1, 14), 59180)])
        with self.assertRaises(ValueError):
            scheduler.parse_race_result("202606040601", "<html></html>")

    def test_parse_suggestion_combo(self):
        self.assertEqual(scheduler.parse_suggestion_combo("単勝", "3 馬名"), (3,))
        self.assertEqual(scheduler.parse_suggestion_combo("複勝", "14 馬名"), (14,))
        self.assertEqual(scheduler.parse_suggestion_combo("枠連", "枠2-枠1"), (1, 2))
        self.assertEqual(scheduler.parse_suggestion_combo("馬連", "10-2"), (2, 10))
        self.assertEqual(scheduler.parse_suggestion_combo("ワイド", "3-5"), (3, 5))
        self.assertEqual(scheduler.parse_suggestion_combo("馬単", "2→1"), (2, 1))
        self.assertEqual(scheduler.parse_suggestion_combo("3連複", "14-2-1"), (1, 2, 14))
        self.assertEqual(scheduler.parse_suggestion_combo("3連単", "2→1→14"), (2, 1, 14))
        for bet_type, combo in [("馬連", "1→2"), ("単勝", "馬名"), ("不明", "1-2"), ("馬単", "1-2")]:
            self.assertIsNone(scheduler.parse_suggestion_combo(bet_type, combo))

    def test_evaluate_bets(self):
        payouts = scheduler.parse_race_result("202606040601", self.PAYOUT_HTML)
        bets = [{"式別": "馬連", "買い目": "2-1"}, {"式別": "単勝", "買い目": "1 馬"},
                {"式別": "3連単", "買い目": "2→1→14"}, {"式別": "馬単", "買い目": "1→2"}]
        result = scheduler.evaluate_bets(bets, payouts)
        self.assertEqual(result["points"], 4)
        self.assertEqual(result["bet"], 400)
        self.assertEqual(result["hit_count"], 2)
        self.assertEqual(result["return"], 7210 + 59180)

    def test_summarize_day(self):
        self.save_bets(self.race1, [{"式別": "馬連", "買い目": "1-2"}, {"式別": "単勝", "買い目": "3 馬"}])
        self.save_bets(self.race2, [{"式別": "複勝", "買い目": "5 馬"}])
        payouts = {self.race1.race_id: {"馬連": [((1, 2), 250)]},
                   self.race2.race_id: {"複勝": [((1,), 130)]}}
        with patch.object(scheduler, "fetch_race_result", side_effect=lambda rid, session: payouts[rid]), \
                patch.object(scheduler, "send_discord") as send:
            scheduler.summarize_day([self.race1, self.race2], self.webhook, self.base, sleep=lambda _: None)
            message = send.call_args.args[1]
            self.assertIn("本日の回収率 (2026/09/20)", message)
            self.assertIn("購入: 3点 300円 / 的中: 1点", message)
            self.assertIn("払戻: 250円 / 収支: -50円", message)
            self.assertIn("回収率: 83.3%", message)
            self.assertIn("中山 1R 馬連 1-2 → 250円", message)
            self.assertNotIn("集計対象外", message)
            scheduler.summarize_day([self.race1, self.race2], self.webhook, self.base, sleep=lambda _: None)
            send.assert_called_once()
        self.assertTrue(json.loads(self.summary_path.read_text(encoding="utf-8"))["done"])

    def test_summarize_day_missing_result(self):
        self.save_bets(self.race1, [{"式別": "馬連", "買い目": "1-2"}])
        self.save_bets(self.race2, [{"式別": "複勝", "買い目": "5 馬"}])

        def fetch(race_id, session):
            if race_id == self.race2.race_id:
                raise ValueError("結果未確定")
            return {"馬連": [((1, 2), 250)]}
        # 期限(最終発走+60分)を過ぎているため1巡で諦める
        with patch.object(scheduler, "fetch_race_result", side_effect=fetch), \
                patch.object(scheduler, "send_discord") as send:
            scheduler.summarize_day([self.race1, self.race2], self.webhook, self.base, sleep=lambda _: None)
            message = send.call_args.args[1]
            self.assertIn("購入: 1点 100円 / 的中: 1点", message)
            self.assertIn("回収率: 250.0%", message)
            self.assertIn("集計対象外: 中山 12R", message)

    def test_summarize_day_no_bets(self):
        self.save_bets(self.race1, [])
        with patch.object(scheduler, "fetch_race_result") as fetch, patch.object(scheduler, "send_discord") as send:
            scheduler.summarize_day([self.race1], self.webhook, self.base, sleep=lambda _: None)
            fetch.assert_not_called()
            self.assertIn("本日の買い目はありませんでした", send.call_args.args[1])

    def test_summarize_day_already_done(self):
        scheduler.save_state(self.summary_path, {"done": True})
        with patch.object(scheduler, "fetch_race_result") as fetch, patch.object(scheduler, "send_discord") as send:
            scheduler.summarize_day([self.race1], self.webhook, self.base, sleep=lambda _: None)
            fetch.assert_not_called()
            send.assert_not_called()

    def test_process_race_stores_bets(self):
        ScheduleTests.predictions(self)
        odds = {"tanpuku": pd.DataFrame([{"馬番": 1, "単勝オッズ": 50, "複勝オッズ下限": 2}])}
        with patch.object(scheduler, "fetch_live_race_odds", return_value=odds), patch.object(scheduler, "send_discord"):
            scheduler.process_race(self.race1, self.webhook, self.base,
                                   lambda: self.race1.start - timedelta(minutes=5))
        state = json.loads((self.base / "suggestions" / "discord_state" / f"{self.race1.race_id}.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual(state["bets"], [{"式別": "複勝", "買い目": "1 馬1", "的中確率(推定)": 0.7,
                                          "オッズ": 2.0, "期待値": 1.4,
                                          "備考": "複勝オッズは下限値で評価（7頭以下のため2着までが的中）"}])


class SchedulePostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.webhook = "https://discord.com/api/webhooks/123/secret"
        self.state_path = self.base / "suggestions" / "discord_state" / "schedule_20260920.json"

    def race(self, race_id, name, start):
        return scheduler.Race(race_id, name, start)

    def races(self):
        return [
            self.race("202607040703", "レースC", datetime(2026, 9, 20, 12, 30, tzinfo=scheduler.JST)),
            self.race("202606040601", "レースA", datetime(2026, 9, 20, 10, 0, tzinfo=scheduler.JST)),
            self.race("202606040612", "レースB", datetime(2026, 9, 20, 16, 0, tzinfo=scheduler.JST)),
        ]

    def test_build_schedule_messages_order_and_format(self):
        messages = scheduler.build_schedule_messages(self.races())
        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertTrue(message.startswith("_ _\n本日のスケジュール (2026/09/20)\n```"))
        self.assertTrue(message.endswith("\n```"))
        self.assertEqual(message.count("```"), 2)
        lines = message.splitlines()
        body = [line for line in lines if "レース" in line]
        self.assertEqual(body, ["10:00 中山  1R レースA",
                                "12:30 中京  3R レースC",
                                "16:00 中山 12R レースB"])

    def test_build_schedule_messages_split(self):
        races = [self.race(f"20260604{1 + i // 12:02d}{1 + i % 12:02d}", f"テストレース{i}" + "あ" * 30,
                           datetime(2026, 9, 20, 10, 0, tzinfo=scheduler.JST) + timedelta(minutes=5 * i))
                 for i in range(100)]
        messages = scheduler.build_schedule_messages(races)
        self.assertGreater(len(messages), 1)
        for message in messages:
            self.assertLessEqual(len(message), 1900)
            self.assertEqual(message.count("```"), 2)
            self.assertIn("本日のスケジュール (2026/09/20)", message)
        self.assertEqual(sum(message.count("テストレース") for message in messages), 100)

    def test_post_schedule_and_skip_when_done(self):
        with patch.object(scheduler, "send_discord") as send:
            scheduler.post_schedule(self.races(), self.webhook, self.base)
            self.assertEqual(send.call_count, 1)
            scheduler.post_schedule(self.races(), self.webhook, self.base)
            send.assert_called_once()
        self.assertTrue(json.loads(self.state_path.read_text(encoding="utf-8"))["done"])

    def test_post_schedule_no_races(self):
        with patch.object(scheduler, "send_discord") as send:
            scheduler.post_schedule([], self.webhook, self.base)
            send.assert_not_called()
        self.assertFalse(self.state_path.exists())


if __name__ == "__main__":
    unittest.main()
