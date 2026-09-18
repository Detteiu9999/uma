import json
import os
import re
import tempfile
import unittest
from datetime import date
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd

import predict_model as pm
import uma


class BlendTests(unittest.TestCase):
    def test_batch_matches_single_races(self):
        groups = np.array(["a", "b", "a", "b", "b", "c"])
        base = np.array([1, 10, 4, 2, -3, 8.0])
        for optional, is_win in ((np.array([9, 2, -1, 4, 7, 0.0]), False),
                                 (np.array([0, 1, .8, .2, .1, .5]), True)):
            for weight in (0, .25, .8):
                with self.subTest(win=is_win, weight=weight):
                    batch = pm.blend_group_scores(base, groups, optional, weight, is_win)
                    for group in pd.unique(groups):
                        mask = groups == group
                        single = pm.blend_group_scores(base[mask], groups[mask],
                                                       optional[mask], weight, is_win)
                        np.testing.assert_array_equal(batch[mask], single)

    def test_constant_scores(self):
        mixed = pm.blend_group_scores([5, 5, 9], ["a", "a", "b"], [2, 2, 3], .4)
        np.testing.assert_array_equal(mixed, [0, 0, 0])
        np.testing.assert_allclose(pm.pl_topk_probs(mixed[:2], 1)[0], [.5, .5])
        np.testing.assert_array_equal(pm.blend_group_scores([5, 5], [0, 0]), [5, 5])

    def test_selected_scores_are_recalibrated(self):
        base = np.array([3., 1., 0., 3., 1., 0.])
        groups = np.repeat(["a", "b"], 3)
        finish = np.array([2, 1, 3, 2, 1, 3])
        lgb = np.array([0., 5., 1., 0., 5., 1.])
        win = np.array([.1, .8, .1, .1, .8, .1])
        with patch.object(pm, "fit_temperature", wraps=pm.fit_temperature) as fit:
            name, weight, scores, temperature, hit = pm.select_blend(
                base, finish, groups, [("lgb", lgb, [.8]), ("win", win, [.8])])
        self.assertEqual(name, "lgb")
        self.assertEqual(hit, 1)
        np.testing.assert_array_equal(scores, pm.blend_group_scores(base, groups, lgb, weight))
        np.testing.assert_array_equal(fit.call_args.args[0], scores)
        self.assertEqual(temperature, pm.fit_temperature(scores, finish, groups)[0])
        self.assertFalse(np.array_equal(scores, base))

    def test_normalized_probabilities(self):
        for n in (1, 2, 3, 5, 18):
            for scale in (0., 1., 1000.):
                scores = np.linspace(-scale, scale, n)
                probabilities = pm.pl_topk_probs(scores, .1)
                for k, p in enumerate(probabilities, 1):
                    self.assertTrue(np.isfinite(p).all())
                    self.assertTrue(((p >= 0) & (p <= 1)).all())
                    self.assertAlmostEqual(p.sum(), min(k, n))
                self.assertTrue((probabilities[0] <= probabilities[1] + 1e-12).all())
                self.assertTrue((probabilities[1] <= probabilities[2] + 1e-12).all())
                np.testing.assert_allclose(probabilities[0], pm.softmax_by_group(scores, [0] * n, .1))


class MetadataTests(unittest.TestCase):
    def test_old_mixed_metadata_rejected(self):
        for flags in ({"has_lgb": True}, {"has_mt": True}, {"has_lgb": True, "has_mt": True}):
            meta = dict(version=7, temperature=1., **flags)
            with self.subTest(flags=flags), \
                    patch("builtins.open", mock_open(read_data=json.dumps(meta))), \
                    patch.object(pm.os.path, "exists", return_value=True), \
                    patch.object(pm.xgb, "Booster") as booster:
                with self.assertRaisesRegex(SystemExit, "再学習"):
                    pm.load_models()
                booster.assert_not_called()

    def test_missing_optional_file_rejected_before_loading(self):
        for flag, weight, filename in (("has_lgb", "ensemble_weight", pm.LGB_MODEL_FILE),
                                       ("has_mt", "mt_weight", pm.MT_MODEL_FILE)):
            meta = {"version": 8, "temperature": 1., flag: True, weight: .3}
            with self.subTest(flag=flag), \
                    patch("builtins.open", mock_open(read_data=json.dumps(meta))), \
                    patch.object(pm.os.path, "exists", return_value=True), \
                    patch.object(pm.os.path, "isfile",
                                 side_effect=lambda p: os.path.basename(p) != filename), \
                    patch.object(pm.xgb, "Booster") as booster:
                with self.assertRaisesRegex(SystemExit, filename):
                    pm.load_models()
                booster.assert_not_called()

    def test_inconsistent_optional_metadata_rejected(self):
        for flag, weight in (("has_lgb", "ensemble_weight"), ("has_mt", "mt_weight")):
            for meta in ({"version": 8, "temperature": 1., flag: True, weight: 0.},
                         {"version": 8, "temperature": 1., flag: True, weight: 1.2},
                         {"version": 8, "temperature": 0., flag: False}):
                with self.subTest(meta=meta), \
                        patch("builtins.open", mock_open(read_data=json.dumps(meta))), \
                        patch.object(pm.os.path, "exists", return_value=True), \
                        patch.object(pm.xgb, "Booster") as booster:
                    with self.assertRaises(SystemExit):
                        pm.load_models()
                    booster.assert_not_called()


class ParsingTests(unittest.TestCase):
    def test_date_place_weather(self):
        cases = {
            "08/02小晴": ("08/02", "小", "晴"),
            "08/02小曇": ("08/02", "小", "曇"),
            "08/02小雨": ("08/02", "小", "雨"),
            "08/02小小": ("08/02", "小", "小"),
            "08/02小小雨": ("08/02", "小", "小雨"),
            "08/02小雪": ("08/02", "小", "雪"),
            "08/02小": ("08/02", "小", ""),
            "08/02中曇": ("08/02", "中", "曇"),
            "08/02東小": ("08/02", "東", "小"),
            "08/02東小雨": ("08/02", "東", "小雨"),
            "08/02": ("08/02", "", ""),
            "": ("", "", ""),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(uma.split_date_place_weather(text), expected)
        past_date, _, _ = uma.split_date_place_weather("08/02小晴")
        self.assertEqual(uma.calc_days_between(date(2025, 8, 30), past_date), 28)

    def test_weight_change_rank(self):
        cases = {
            "480": ("480", "", ""),
            "554(+4)1": ("554", "+4", "1"),
            "480(-2)12": ("480", "-2", "12"),
            "480(0)": ("480", "0", ""),
            "480()3": ("480", "", "3"),
            "計不": ("計不", "", ""),
            "": ("", "", ""),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(uma.split_weight_change_3f(text), expected)


class FeatureTests(unittest.TestCase):
    def test_class_change_matches_separate_races(self):
        frame = pd.DataFrame({
            "URLコード": ["2501010101", "2501010102"] * 3,
            "通算_全": ["0-0-0-3", "3-0-0-3"] * 3,
            "前走のレース名": ["未勝利", "2勝"] * 3,
            "着順": [1, 1, 2, 2, 3, 3],
        }, index=[11, 8, 5, 3, 9, 7])
        for snapshot in ("pre", "post"):
            with self.subTest(snapshot=snapshot), patch.object(pm, "CAREER_SNAPSHOT", snapshot):
                batch = pm.build_features(frame)["クラス変化"]
                individual = pd.concat([
                    pm.build_features(race)["クラス変化"]
                    for _, race in frame.groupby("URLコード")
                ]).reindex(frame.index)
                pd.testing.assert_series_equal(batch, individual)
                np.testing.assert_array_equal(batch, [0., 1.] * 3)
                changed = frame.copy()
                changed.loc[changed["URLコード"] == "2501010102", "通算_全"] = "8-0-0-3"
                other = pm.build_features(changed)["クラス変化"]
                mask = frame["URLコード"] == "2501010101"
                pd.testing.assert_series_equal(batch[mask], other[mask])

    def test_career_pace_and_raw_result_exclusion(self):
        frame = pd.DataFrame({
            "URLコード": ["2601010101"] * 4, "馬番": [1, 2, 3, 4],
            "枠番": [1, 1, 2, 3], "距離": [1600] * 4,
            pm.STYLE_COL: ["01000000", "00010000", "00000100", "00000001"],
            "通算_全": ["1-2-3-4"] * 4, "前走のレース名": ["1勝"] * 4,
            "脚質": ["逃", "先", "差", "追"], "着順": [1, 2, 3, 4],
            "通過順位": ["01010101"] * 4, "スピード指数": [99] * 4,
        })
        with patch.object(pm, "CAREER_SNAPSHOT", "post"):
            features = pm.build_features(frame)
        self.assertTrue(set(pm.PACE_FEATURE_NAMES).issubset(features.columns))
        dropped = pm.drop_career_features(features)
        self.assertFalse(set(pm.PACE_FEATURE_NAMES) & set(dropped.columns))
        self.assertFalse(any(pm.CAREER_FEATURE_PATTERN.search(c) for c in dropped.columns))
        self.assertFalse(set(pm.PACE_FEATURE_NAMES) & set(pm.drop_pace_features(features).columns))
        for name in ("脚質_数値", "脚質", "着順", "通過順位", "スピード指数"):
            self.assertNotIn(name, features.columns)
        changed = frame.copy()
        changed["着順"] = [4, 3, 2, 1]
        changed["脚質"] = ["追", "差", "先", "逃"]
        with patch.object(pm, "CAREER_SNAPSHOT", "post"):
            pd.testing.assert_frame_equal(dropped, pm.drop_career_features(pm.build_features(changed)))
        self.assertIn("枠番", dropped.columns)
        self.assertFalse(pm.USE_PACE_FEATURES)


class PredictionOutputTests(unittest.TestCase):
    def _race_frame(self, code):
        rows = 4
        return pd.DataFrame({
            "URLコード": [code] * rows, "枠番": ["1", "1", "2", "3"],
            "馬番": ["1", "2", "3", "4"], "距離": ["1600"] * rows,
            "馬名": ["a", "b", "c", "d"], "芝orダート": ["芝"] * rows, "競馬場": ["札幌"] * rows,
            "性別": ["牝"] * rows, "年齢": ["4"] * rows, "斤量": ["55"] * rows,
            "馬体重": ["480"] * rows, "体重増減": ["0"] * rows, "馬場": ["良"] * rows,
            "調教師": [""] * rows, "騎手": [""] * rows,
            pm.STYLE_COL: ["01000000", "00010000", "00000100", "00000001"],
            "通算_全": ["0-0-0-2"] * rows,
            "前走のスピード指数": ["60"] * rows, "前走の着順": ["1"] * rows,
            "前走の距離": ["1600"] * rows, "前走の頭数": ["16"] * rows,
            "前走の芝orダート": ["芝"] * rows, "前走の競馬場": ["札"] * rows,
            "前走からの日数": ["30"] * rows, "前走の通過順位": ["01010101"] * rows,
            "前走のレース名": ["1勝"] * rows, "前走の人気": ["3"] * rows,
        })

    def _model_mocks(self, frame):
        base = np.linspace(1., 2., len(frame))
        model = Mock()
        model.predict.return_value = base
        return model, base

    def _identity_features(self, feat_df, orig_df, stats):
        return feat_df

    def _run_single_race(self, tmp, frame, code):
        model, _ = self._model_mocks(frame)
        with patch.object(pm, "OUT_DIR", tmp), \
                patch.object(pm, "PRED_DIR", tmp), \
                patch.object(pm.glob, "glob", return_value=["r.csv"]), \
                patch.object(pm.pd, "read_csv", return_value=frame), \
                patch.object(pm, "normalize_career_columns", side_effect=lambda df: df), \
                patch.object(pm, "add_personnel_features", self._identity_features), \
                patch.object(pm, "fit_category_maps", return_value={}), \
                patch.object(pm, "apply_category_maps",
                             side_effect=lambda X, maps: (X.copy(), [])):
            pm.run_prediction(model, {}, ["枠番", "馬番"], [], {}, 1.0,
                              race_code=code, output_dir=tmp)

    def test_single_race_output_and_raw_results_excluded(self):
        code = "2601010101"
        frame = self._race_frame(code)
        frame["着順"] = ["1", "2", "3", "4"]
        with tempfile.TemporaryDirectory() as tmp:
            self._run_single_race(tmp, frame, code)
            files = os.listdir(tmp)
            self.assertEqual(sorted(f for f in files if f.startswith("pred_")),
                             ["pred_2026_第1回札幌1日目_1R.csv"])
            self.assertIn("_all_predictions.csv", files)
            output = pd.read_csv(os.path.join(tmp, "pred_2026_第1回札幌1日目_1R.csv"),
                                 encoding="utf-8-sig")
        self.assertEqual(sorted(output["枠番"].tolist()), [1, 1, 2, 3])
        self.assertEqual(set(output["URLコード"]), {int(code)})
        self.assertNotIn("着順", output.columns)
        probabilities = output["1着確率"].to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(probabilities.sum(), 1)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertEqual(len(output), len(frame))
        scores = output["スコア"].to_numpy()
        np.testing.assert_array_equal(scores, np.sort(scores)[::-1])

    def test_multiple_codes_rejected(self):
        frame = self._race_frame("2601010101")
        frame.loc[2:, "URLコード"] = "2601010102"
        model, _ = self._model_mocks(frame)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pm, "OUT_DIR", tmp), \
                patch.object(pm, "PRED_DIR", tmp), \
                patch.object(pm.glob, "glob", return_value=["r.csv"]), \
                patch.object(pm.pd, "read_csv", return_value=frame), \
                patch.object(pm, "normalize_career_columns", side_effect=lambda df: df), \
                patch.object(pm, "add_personnel_features", side_effect=lambda X, *_: X), \
                patch.object(pm, "fit_category_maps", return_value={}), \
                patch.object(pm, "apply_category_maps", side_effect=lambda X, maps: (X.copy(), [])), \
                self.assertRaisesRegex(SystemExit, re.escape("2601010102")):
            pm.run_prediction(model, {}, ["枠番", "馬番"], [], {}, 1.0, output_dir=tmp)


if __name__ == "__main__":
    unittest.main()
