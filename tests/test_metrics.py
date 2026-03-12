import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from eval.metrics import eer, fnmr_at_fmr, roc_curve, summarize_verification, tar_at_far


class MetricsTest(unittest.TestCase):
    def test_tar_at_far_picks_best_tar_under_cap(self) -> None:
        scores = np.array([0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.10, 0.05, 0.01], dtype=np.float32)
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
        tar, threshold = tar_at_far(scores, labels, 0.5)
        self.assertAlmostEqual(tar, 1.0)
        self.assertAlmostEqual(threshold, 0.80, places=5)

    def test_roc_curve_is_monotonic(self) -> None:
        scores = np.array([0.9, 0.75, 0.7, 0.6, 0.4, 0.2], dtype=np.float32)
        labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
        _, fprs, tprs = roc_curve(scores, labels)
        self.assertTrue(np.all(np.diff(fprs) >= -1e-12))
        self.assertTrue(np.all(np.diff(tprs) >= -1e-12))

    def test_tied_scores_do_not_break_summary(self) -> None:
        scores = np.array([0.8, 0.8, 0.6, 0.6, 0.4, 0.4], dtype=np.float32)
        labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
        summary = summarize_verification(scores, labels, bootstrap_samples=0)
        self.assertIn("eer", summary)
        self.assertIn("far_targets", summary)
        fnmr, threshold = fnmr_at_fmr(scores, labels, 0.5)
        self.assertGreaterEqual(fnmr, 0.0)
        self.assertLessEqual(fnmr, 1.0)
        self.assertTrue(np.isfinite(threshold))

    def test_eer_is_bounded(self) -> None:
        scores = np.array([0.95, 0.92, 0.2, 0.1], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=np.int32)
        value, threshold = eer(scores, labels)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertTrue(np.isfinite(threshold))


if __name__ == "__main__":
    unittest.main()
