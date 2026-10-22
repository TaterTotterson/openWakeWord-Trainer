#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibrate_model import (  # noqa: E402
    choose_confirmation_operating_point,
    choose_confirmation_threshold,
    choose_operating_point,
    choose_threshold,
    deterministic_sample,
)


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

    def test_multi_patience_selects_lowest_safe_patience(self) -> None:
        metrics = [
            {"threshold": 0.99, "patience": 2, "false_positive_rate": 0.02, "positive_recall": 0.99},
            {"threshold": 0.98, "patience": 3, "false_positive_rate": 0.0, "positive_recall": 0.94},
            {"threshold": 0.95, "patience": 4, "false_positive_rate": 0.0, "positive_recall": 0.97},
        ]
        selected, reason = choose_operating_point(
            metrics,
            max_false_positive_rate=0.0,
            min_positive_recall=0.90,
            fallback_threshold=0.95,
        )
        self.assertEqual((selected["threshold"], selected["patience"]), (0.98, 3))
        self.assertEqual(reason, "meets_false_positive_and_recall_targets")

    def test_maximum_threshold_is_a_valid_measured_operating_point(self) -> None:
        selected, _ = choose_operating_point(
            [{"threshold": 0.99, "patience": 5, "false_positive_rate": 0.0, "positive_recall": 0.90}],
            max_false_positive_rate=0.0,
            min_positive_recall=0.90,
            fallback_threshold=0.95,
        )
        self.assertEqual((selected["threshold"], selected["patience"]), (0.99, 5))

    def test_confirmation_prefers_low_latency_then_high_threshold(self) -> None:
        selected, _ = choose_confirmation_operating_point(
            [
                {"threshold": 0.95, "patience": 2, "positive_recall": 0.90},
                {"threshold": 0.99, "patience": 2, "positive_recall": 0.82},
                {"threshold": 0.99, "patience": 3, "positive_recall": 0.91},
            ],
            min_positive_recall=0.80,
            fallback_threshold=0.90,
        )
        self.assertEqual((selected["threshold"], selected["patience"]), (0.99, 2))

    def test_corrective_partitions_are_deterministic_and_disjoint(self) -> None:
        paths = [Path(f"sample-{index}.wav") for index in range(100)]
        first = deterministic_sample(paths, 20, partition=0)
        second = deterministic_sample(paths, 20, partition=1)
        self.assertEqual(first, deterministic_sample(paths, 20, partition=0))
        self.assertTrue(set(first).isdisjoint(second))


if __name__ == "__main__":
    unittest.main()
