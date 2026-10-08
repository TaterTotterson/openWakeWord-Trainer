#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibrate_model import choose_confirmation_threshold, choose_threshold  # noqa: E402


class CalibrationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = [
            {"threshold": 0.90, "false_positive_rate": 0.004, "positive_recall": 0.815},
            {"threshold": 0.91, "false_positive_rate": 0.002, "positive_recall": 0.80},
            {"threshold": 0.92, "false_positive_rate": 0.002, "positive_recall": 0.80},
            {"threshold": 0.95, "false_positive_rate": 0.002, "positive_recall": 0.715},
            {"threshold": 0.99, "false_positive_rate": 0.0, "positive_recall": 0.38},
        ]

    def test_standalone_preserves_strict_zero_false_positive_policy(self) -> None:
        threshold, reason = choose_threshold(
            self.metrics,
            max_false_positive_rate=0.0,
            min_positive_recall=0.70,
            fallback_threshold=0.95,
        )
        self.assertEqual(threshold, 0.99)
        self.assertEqual(reason, "meets_false_positive_target_but_recall_is_low")

    def test_dual_confirmation_preserves_more_genuine_wakes(self) -> None:
        threshold, reason = choose_confirmation_threshold(
            self.metrics,
            min_positive_recall=0.80,
            fallback_threshold=0.90,
        )
        self.assertEqual(threshold, 0.92)
        self.assertEqual(reason, "meets_dual_confirmation_recall_target")

    def test_confirmation_falls_back_when_validation_cannot_meet_target(self) -> None:
        threshold, reason = choose_confirmation_threshold(
            [{"threshold": 0.99, "false_positive_rate": 0.0, "positive_recall": 0.3}],
            min_positive_recall=0.80,
            fallback_threshold=0.90,
        )
        self.assertEqual(threshold, 0.90)
        self.assertEqual(reason, "no_threshold_met_dual_confirmation_recall_target")


if __name__ == "__main__":
    unittest.main()
