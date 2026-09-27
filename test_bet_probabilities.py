import io
import math
import unittest
from itertools import combinations, combinations_with_replacement, permutations
from unittest.mock import patch

import numpy as np
import pandas as pd

import check_results as results
import suggest_bets as bets


def make_horses(probabilities, frames=None):
    if frames is None:
        frames = list(range(1, len(probabilities) + 1))
    return [bets.Horse({
        "馬番": i + 1, "馬名": str(i + 1), "枠番": frames[i],
        "1着確率": p, "2着以内確率": 0.5, "3着以内確率": 0.75,
    }) for i, p in enumerate(probabilities)]


class HarvilleTests(unittest.TestCase):
    def test_total_probability(self):
        for p in ([0.4, 0.3, 0.2, 0.1], [0.25] * 4,
                  [0.5, 0.3, 0.2, 0], [0.9999999999999998, 1e-16, 1e-16]):
            with self.subTest(p=p):
                indices = range(len(p))
                for function, orders in (
                    (bets.harville_exacta, permutations(indices, 2)),
                    (bets.harville_quinella, combinations(indices, 2)),
                    (bets.harville_trifecta, permutations(indices, 3)),
                    (bets.harville_trio, combinations(indices, 3)),
                ):
                    self.assertAlmostEqual(math.fsum(function(p, *o) for o in orders), 1)

    def test_marginals(self):
        p = [0.4, 0.3, 0.2, 0.1]
        orders = {o: bets.harville_trifecta(p, *o) for o in permutations(range(4), 3)}
        for a in range(4):
            self.assertAlmostEqual(sum(bets.harville_exacta(p, a, b)
                                       for b in range(4) if b != a), p[a])
            self.assertAlmostEqual(sum(v for o, v in orders.items() if o[0] == a), p[a])
        for a, b in permutations(range(4), 2):
            self.assertAlmostEqual(sum(v for o, v in orders.items() if o[:2] == (a, b)),
                                   bets.harville_exacta(p, a, b))
            self.assertAlmostEqual(sum(v for o, v in orders.items() if a in o and b in o),
                                   bets.harville_wide(p, a, b))
        for trio in combinations(range(4), 3):
            self.assertAlmostEqual(sum(v for o, v in orders.items() if set(o) == set(trio)),
                                   bets.harville_trio(p, *trio))

    def test_uniform(self):
        for n in (3, 4, 8):
            p = [1 / n] * n
            self.assertAlmostEqual(bets.harville_exacta(p, 0, 1), 1 / (n * (n - 1)))
            self.assertAlmostEqual(bets.harville_quinella(p, 0, 1), 2 / (n * (n - 1)))
            self.assertAlmostEqual(bets.harville_trifecta(p, 0, 1, 2),
                                   1 / (n * (n - 1) * (n - 2)))
            self.assertAlmostEqual(bets.harville_trio(p, 0, 1, 2), 1 / math.comb(n, 3))
            self.assertAlmostEqual(bets.harville_wide(p, 0, 1), 6 / (n * (n - 1)))

    def test_small_fields_and_zero_support(self):
        self.assertAlmostEqual(bets.harville_exacta([0.7, 0.3], 0, 1), 0.7)
        self.assertAlmostEqual(bets.harville_quinella([0.7, 0.3], 0, 1), 1)
        self.assertAlmostEqual(bets.harville_wide([0.7, 0.3], 0, 1), 1)
        self.assertAlmostEqual(bets.harville_trio([0.5, 0.3, 0.2], 0, 1, 2), 1)
        self.assertEqual(bets.harville_exacta([1, 0], 0, 1), 0)
        self.assertEqual(bets.harville_trifecta([0.5, 0.5, 0], 0, 1, 2), 0)
        with self.assertRaises(ValueError):
            bets.harville_exacta([1], 0, 1)
        with self.assertRaises(ValueError):
            bets.harville_trifecta([0.5, 0.5], 0, 1, 2)

    def test_stable_remaining_denominators(self):
        p = [1.0, 1e-200, 1e-200]
        self.assertAlmostEqual(bets.harville_exacta(p, 0, 1), 0.5)
        self.assertAlmostEqual(bets.harville_trifecta(p, 0, 1, 2), 0.5)

    def test_duplicates_and_invalid_indices(self):
        p = [0.4, 0.3, 0.2, 0.1]
        for function, size in ((bets.harville_exacta, 2), (bets.harville_quinella, 2),
                               (bets.harville_wide, 2), (bets.harville_trifecta, 3),
                               (bets.harville_trio, 3)):
            for indices in ([0] * size, [-1] + list(range(1, size)),
                            [4] + list(range(1, size)), [0.0] + list(range(1, size)),
                            [True] + list(range(1, size))):
                with self.subTest(function=function.__name__, indices=indices):
                    with self.assertRaises(ValueError):
                        function(p, *indices)

    def test_invalid_probabilities(self):
        invalid = ([], [0, 0, 0], [-0.1, 0.5, 0.6], [1.1, 0, 0],
                   [float("nan"), 0.3, 0.7], [float("inf"), 0, 0],
                   [float("-inf"), 0, 0], [0.1, 0.1, 0.1],
                   [[0.3, 0.3, 0.4]], ["bad", 0.3, 0.7])
        for p in invalid:
            for function, indices in ((bets.harville_exacta, (0, 1)),
                                      (bets.harville_quinella, (0, 1)),
                                      (bets.harville_wide, (0, 1)),
                                      (bets.harville_trifecta, (0, 1, 2)),
                                      (bets.harville_trio, (0, 1, 2))):
                with self.subTest(p=p, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(p, *indices)
            with self.assertRaises(ValueError):
                bets.harville_wakuren(p, [1, 2, 3], 1, 2)

    def test_win_vector_validation_and_normalization(self):
        np.testing.assert_allclose(bets.win_probs_vector(make_horses([0.2, 0.3])), [0.4, 0.6])
        np.testing.assert_allclose(bets.win_probs_vector(make_horses([1e-320, 1e-320])), [0.5, 0.5])
        for p in ([], [0, 0], [-0.1, 0.5], [1.1, 0.2], [float("nan"), 0.5],
                  [float("inf"), 0.5]):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    bets.win_probs_vector(make_horses(p))

    def test_frames_partition_and_whole_field(self):
        p = [0.4, 0.3, 0.2, 0.1]
        frames = [1, 1, 2, 3]
        total = 0
        for w1, w2 in combinations_with_replacement((1, 2, 3), 2):
            expected = sum(bets.harville_exacta(p, a, b)
                           for a, b in permutations(range(4), 2)
                           if sorted((frames[a], frames[b])) == sorted((w1, w2)))
            actual = bets.harville_wakuren(p, frames, w1, w2)
            self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(actual, bets.harville_wakuren(p, frames, w2, w1))
            total += actual
        self.assertAlmostEqual(total, 1)
        self.assertEqual(bets.harville_wakuren(p, frames, 2, 2), 0)
        self.assertEqual(bets.harville_wakuren(p, frames, 1, 8), 0)
        self.assertAlmostEqual(bets.harville_wakuren([0.25] * 4, [1, 2, 3, 4], 1, 2), 1 / 6)
        self.assertAlmostEqual(bets.harville_wakuren(p, [1] * 4, 1, 1), 1)
        with self.assertRaises(ValueError):
            bets.harville_wakuren(p, [1], 1, 1)


class SuggestionTests(unittest.TestCase):
    def test_frame_csv_round_trip(self):
        horses = make_horses([0.25] * 4, [1, 1, 2, 3])
        odds = pd.read_csv(io.StringIO("枠番1,枠番2,枠連オッズ\n1,1,10\n1,2,10\n2,3,10\n"))
        rows = bets.suggest_for_race(horses, {"wakuren": odds}, 0)
        self.assertEqual({r["買い目"]: r["的中確率(推定)"] for r in rows},
                         {"枠1-枠1": 0.1667, "枠1-枠2": 0.3333, "枠2-枠3": 0.1667})
        df = pd.DataFrame(rows)
        df["買い目"] = df["買い目"].map(lambda x: f'="{x}"')
        loaded = pd.read_csv(io.StringIO(df.to_csv(index=False)))
        self.assertEqual([results.parse_combo(r["式別"], r["買い目"])
                          for _, r in loaded.iterrows()], [(1, 1), (1, 2), (2, 3)])

    def test_isolated_output_cli(self):
        horses = make_horses([0.25] * 4, [1, 1, 2, 3])
        prediction = pd.DataFrame([{
            "馬番": h.num, "馬名": h.name, "枠番": h.waku,
            "1着確率": h.p1, "2着以内確率": h.p2, "3着以内確率": h.p3,
        } for h in horses])
        odds = pd.DataFrame([{"枠番1": 1, "枠番2": 1, "枠連オッズ": 10}])
        for flags, expected in (([], 0.1667), (["--indep"], 1.0)):
            with self.subTest(flags=flags), patch("sys.argv", [
                    "suggest_bets.py", "--place", "札幌", "--race", "11",
                    "--output", "isolated.csv", *flags]), \
                    patch.object(bets, "load_predictions", return_value={
                        ("札幌", 2, 6, 11): prediction, ("札幌", 2, 6, 12): prediction}), \
                    patch.object(bets, "load_odds", return_value={
                        ("20260906_札幌2回6日", 11, "wakuren"): odds,
                        ("20260906_札幌2回6日", 12, "wakuren"): odds}), \
                    patch.object(bets.os, "makedirs"), \
                    patch.object(pd.DataFrame, "to_csv", autospec=True) as write_csv, \
                    patch("sys.stdout", new_callable=io.StringIO):
                bets.main()
                write_csv.assert_called_once()
                frame, path = write_csv.call_args.args
                self.assertEqual(path, "isolated.csv")
                self.assertEqual(frame["レース番号"].tolist(), ["11R"])
                self.assertEqual(frame["買い目"].tolist(), ['="枠1-枠1"'])
                self.assertEqual(frame["的中確率(推定)"].tolist(), [expected])

    def test_independent_comparison(self):
        horses = make_horses([0.4, 0.3, 0.2, 0.1])
        odds = {"umatan": pd.DataFrame([{"馬番1": 1, "馬番2": 2, "馬単オッズ": 10}]),
                "wakuren": pd.DataFrame([{"枠番1": 1, "枠番2": 2, "枠連オッズ": 10}])}
        independent = bets.suggest_for_race(horses, odds, 0, use_harville=False)
        self.assertEqual({r["式別"]: r["的中確率(推定)"] for r in independent},
                         {"馬単": 0.2, "枠連": 0.25})
        harville = bets.suggest_for_race(horses, odds, 0)
        self.assertNotEqual(harville, independent)

    def test_add_rejects_bad_odds_and_probabilities(self):
        for invalid in (float("nan"), float("inf"), float("-inf"), -1, 0, None, "bad"):
            with self.subTest(odds=invalid):
                odds = {"tanpuku": pd.DataFrame([{"馬番": 1, "単勝オッズ": invalid}])}
                self.assertEqual(bets.suggest_for_race(make_horses([0.5, 0.5]), odds, 0), [])
        for invalid in (float("nan"), float("inf"), float("-inf"), -1, 1.1):
            with self.subTest(probability=invalid):
                horses = make_horses([0.5, 0.5])
                horses[0].p2 = invalid  # 7頭以下のため複勝は2着以内確率で評価
                odds = {"tanpuku": pd.DataFrame([{"馬番": 1, "複勝オッズ下限": 10}])}
                self.assertEqual(bets.suggest_for_race(horses, odds, 0), [])
        odds = {"umaren": pd.DataFrame([{"馬番1": 1, "馬番2": 2, "馬連オッズ": 10}])}
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with patch.object(bets, "harville_quinella", return_value=invalid):
                self.assertEqual(bets.suggest_for_race(make_horses([0.5, 0.5]), odds, 0), [])
        with self.assertRaises(ValueError):
            bets.suggest_for_race(make_horses([0, 0]), {}, 0)

    def test_place_field_size_rule(self):
        # 出走7頭以下のレースでは複勝の払い戻し対象は2着まで（2着以内確率で評価）
        odds = {"tanpuku": pd.DataFrame([{"馬番": 1, "複勝オッズ下限": 10}])}
        for field in (7, 8):
            with self.subTest(field=field):
                horses = make_horses([0.4] + [0.1] * (field - 1))
                horses[0].p2, horses[0].p3 = 0.4, 0.7
                (bet,) = bets.suggest_for_race(horses, odds, 0)
                self.assertEqual(bet["式別"], "複勝")
                if field <= 7:
                    self.assertEqual(bet["的中確率(推定)"], 0.4)
                    self.assertIn("2着まで", bet["備考"])
                else:
                    self.assertEqual(bet["的中確率(推定)"], 0.7)
                    self.assertNotIn("2着まで", bet["備考"])
        # 7頭以下では2着以内確率が閾値未満の馬は複勝から除外される
        horses = make_horses([0.4] + [0.1] * 6)
        horses[0].p2, horses[0].p3 = 0.05, 0.9
        self.assertEqual(bets.suggest_for_race(horses, odds, 0), [])
        horses8 = make_horses([0.4] + [0.1] * 7)
        horses8[0].p2, horses8[0].p3 = 0.05, 0.9
        (bet,) = bets.suggest_for_race(horses8, odds, 0)
        self.assertEqual(bet["的中確率(推定)"], 0.9)

    def test_duplicate_horses_and_combinations(self):
        horses = make_horses([0.4, 0.3, 0.3])
        odds = {
            "umaren": pd.DataFrame([{"馬番1": 1, "馬番2": 1, "馬連オッズ": 10}]),
            "wide": pd.DataFrame([{"馬番1": 1, "馬番2": 1, "ワイドオッズ": "10-12"}]),
            "umatan": pd.DataFrame([{"馬番1": 1, "馬番2": 1, "馬単オッズ": 10}]),
            "fuku3": pd.DataFrame([{"馬番1": 1, "馬番2": 2, "馬番3": 1, "3連複オッズ": 10}]),
            "tan3": pd.DataFrame([{"1着馬番": 1, "2着馬番": 2, "3着馬番": 1, "3連単オッズ": 10}]),
        }
        for mode in (True, False):
            self.assertEqual(bets.suggest_for_race(horses, odds, 0, mode), [])
        horses[1].num = horses[0].num
        with self.assertRaises(ValueError):
            bets.suggest_for_race(horses, {}, 0)


class ResultTests(unittest.TestCase):
    def test_plain_and_excel_combos(self):
        for kind, text, expected in (
            ("単勝", "1 Horse", (1,)), ("複勝", "2 Horse", (2,)),
            ("枠連", "枠1-枠1", (1, 1)), ("馬連", "1-2", (1, 2)),
            ("ワイド", "1-3", (1, 3)), ("馬単", "1→2", (1, 2)),
            ("3連複", "1-2-3", (1, 2, 3)), ("3連単", "1→2→3", (1, 2, 3)),
        ):
            for combo in (text, f'="{text}"', f'  ="{text}"  '):
                with self.subTest(kind=kind, combo=combo):
                    self.assertEqual(results.parse_combo(kind, combo), expected)
                    self.assertTrue(results.is_hit(kind, combo, {1: 1, 2: 2, 3: 3},
                                                   {1: 1, 2: 1, 3: 2}))
        for combo in (None, float("nan"), "", '=\"1-2', '=SUM(1,2)', '="bad"'):
            self.assertIsNone(results.parse_combo("馬連", combo))

    def test_place_field_size_result_rule(self):
        # 複勝の的中判定: 出走7頭以下は2着まで、8頭以上は3着まで
        finish7 = {i: i for i in range(1, 8)}
        finish8 = {i: i for i in range(1, 9)}
        self.assertTrue(results.is_hit("複勝", "2 Horse", finish7, {}))
        self.assertFalse(results.is_hit("複勝", "3 Horse", finish7, {}))
        self.assertTrue(results.is_hit("複勝", "3 Horse", finish8, {}))
        # starters 引数で出走頭数を指定できる（中止馬がいても出走頭数ベースで判定）
        self.assertTrue(results.is_hit("複勝", "3 Horse", finish7, {}, starters=8))
        self.assertFalse(results.is_hit("複勝", "3 Horse", finish8, {}, starters=7))

    def test_single_race_cli_filters(self):
        df = pd.DataFrame([{
            "競馬場": place, "回": 1, "日": 1, "レース番号": race,
            "式別": "馬連", "買い目": '="1-2"', "的中確率(推定)": 0.5,
            "期待値": 5, "オッズ": 10,
        } for place, race in (("札幌", "11R"), ("札幌", "12R"), ("東京", "11R"))])
        cases = (([], 3), (["--race", "11"], 2), (["--place", "札幌"], 2),
                 (["--race", "11", "--place", "札幌"], 1),
                 (["--race", "1", "--place", "札幌"], 0))
        for flags, count in cases:
            with self.subTest(flags=flags), patch("sys.argv", ["check_results.py", "--suggestions", "isolated.csv", *flags]), \
                    patch.object(results.os.path, "exists", return_value=True), \
                    patch.object(results, "load_results", return_value={
                        (place, 1, 1, race): {1: 1, 2: 2, 3: 3}
                        for place, race in (("札幌", 11), ("札幌", 12), ("東京", 11))}), \
                    patch.object(results.glob, "glob", return_value=[]), \
                    patch.object(results.pd, "read_csv", return_value=df) as read_csv, \
                    patch("sys.stdout", new_callable=io.StringIO) as output:
                results.main()
                read_csv.assert_called_once_with("isolated.csv", encoding="utf-8-sig")
                self.assertIn(f"対象買い目: {count} 点", output.getvalue())
                self.assertIn(f"的中点数 : {count} 点", output.getvalue())
                self.assertIn("結果未照合 0 点", output.getvalue())


if __name__ == "__main__":
    unittest.main()
