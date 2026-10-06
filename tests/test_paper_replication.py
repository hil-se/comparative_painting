from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/deep_learning"))
sys.path.insert(0, str(ROOT / "code/baseline"))
sys.path.insert(0, str(ROOT / "code/human_survey"))

from manuscript_statistical_tests import exact_wilcoxon, matched_test, holm
from run_sidhu_heldout_ols import impute_training_medians, load_predictors, fit_condition
from human_rating_agreement_unified import create_gt_comparative_column, generate_pairs_comparative


class StatisticalTests(unittest.TestCase):
    def test_exact_distribution_matches_scipy_without_ties(self):
        left = [1., -2., 3., -4., 5., 6., 7., 8., 9., 10.]
        p, delta, nonzero = exact_wilcoxon(left, [0.] * 10)
        self.assertEqual(p, wilcoxon(left, method="exact").pvalue)
        self.assertEqual(nonzero, 10)
        self.assertAlmostEqual(delta, np.mean(left))

    def test_zeros_remain_in_effect_size_and_all_zero_test_is_defined(self):
        self.assertEqual(exact_wilcoxon([0., 2.], [0., 0.]), (1., 1., 1))
        self.assertEqual(exact_wilcoxon([0., 0.], [0., 0.]), (1., 0., 0))

    def test_tied_signed_ranks_match_enumerated_distribution(self):
        # Ranks 1.5, 1.5, 3: only all-positive/all-negative permutations are as extreme.
        self.assertEqual(exact_wilcoxon([1., 1., 2.], [0.] * 3)[0], 2 / 8)

    def test_unmatched_seeds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "ten matched seeds"):
            matched_test("missing", {i: float(i) for i in range(10)}, {i: float(i) for i in range(9)})

    def test_holm_is_monotone_in_sorted_raw_pvalues(self):
        self.assertEqual(holm([.01, .04, .03]), [.03, .06, .06])


class BaselineAndSurveyTests(unittest.TestCase):
    def test_reconstructed_heldout_ols_matches_reported_results(self):
        expected = {("abstract", "beauty"): .339, ("abstract", "liking"): .382,
                    ("representational", "beauty"): .408,
                    ("representational", "liking"): .485}
        for category in ("abstract", "representational"):
            predictors, ids = load_predictors(ROOT, category)
            for target in ("beauty", "liking"):
                rows, _ = fit_condition(ROOT, category, target, predictors, ids, list(range(10)))
                values = [r["spearman"] for r in rows if r["protocol"] == "training_median"]
                self.assertAlmostEqual(float(np.mean(values)), expected[(category, target)], delta=.0005)

    def test_imputation_does_not_use_heldout_predictor_values(self):
        values = np.array([[1., np.nan], [3., 4.], [np.nan, 10000.]])
        imputed = impute_training_medians(values, np.array([0, 1]))
        np.testing.assert_array_equal(imputed, [[1., 4.], [3., 4.], [2., 10000.]])

    def test_representational_beauty_gt_column_names_are_supported(self):
        frame = pd.DataFrame({"GT_A_Beauty": [4., 2., 3.], "GT_B_Beauty": [1., 5., 3.]})
        self.assertEqual(create_gt_comparative_column(frame).GT.tolist(), ["A", "B", None])

    def test_missing_comparative_responses_are_not_counted_as_b(self):
        pairs = generate_pairs_comparative(pd.Series(["A", None, "B"]), pd.Series(["A", "B", "A"]))
        self.assertEqual(pairs["agree"], [True, False])


if __name__ == "__main__":
    unittest.main()
