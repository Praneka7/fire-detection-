"""Unit tests for the AI verification gate (model inference is mocked)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from fire_detector import ConsecutiveFireConfirmation, detect_verified_fire


class FakeClassifier:
    def __init__(self, score: float | None, available: bool = True) -> None:
        self.score = score
        self.available = available
        self.threshold = 0.90
        self.calls = 0

    def predict_region(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> float | None:
        self.calls += 1
        return self.score


def warm_moving_frame(offset: int) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.ellipse(frame, (80 + offset, 120), (36, 58), 0, 0, 360, (0, 105, 255), -1)
    return frame


class VerifiedFireGateTests(unittest.TestCase):
    def test_confirmation_requires_consecutive_verified_frames_and_resets(self) -> None:
        confirmation = ConsecutiveFireConfirmation(3)
        self.assertEqual(confirmation.update(True), (1, False))
        self.assertEqual(confirmation.update(True), (2, False))
        self.assertEqual(confirmation.update(False), (0, False))
        self.assertEqual(confirmation.update(True), (1, False))
        self.assertEqual(confirmation.update(True), (2, False))
        self.assertEqual(confirmation.update(True), (3, True))

    def test_warm_moving_candidate_is_rejected_when_ai_says_non_fire(self) -> None:
        classifier = FakeClassifier(0.12)
        previous_gray = cv2.cvtColor(warm_moving_frame(0), cv2.COLOR_BGR2GRAY)
        _, _, regions, candidate_count = detect_verified_fire(
            warm_moving_frame(10), previous_gray, classifier, min_area=250, confidence_threshold=0.90
        )
        self.assertGreater(candidate_count, 0)
        self.assertEqual(regions, [])
        self.assertGreater(classifier.calls, 0)

    def test_region_is_accepted_only_above_ai_threshold(self) -> None:
        classifier = FakeClassifier(0.97)
        previous_gray = cv2.cvtColor(warm_moving_frame(0), cv2.COLOR_BGR2GRAY)
        _, _, regions, _ = detect_verified_fire(
            warm_moving_frame(10), previous_gray, classifier, min_area=250, confidence_threshold=0.90
        )
        self.assertTrue(regions)
        self.assertGreaterEqual(regions[0][2], 0.90)

    def test_unavailable_ai_model_fails_closed(self) -> None:
        classifier = FakeClassifier(None, available=False)
        previous_gray = cv2.cvtColor(warm_moving_frame(0), cv2.COLOR_BGR2GRAY)
        _, _, regions, candidate_count = detect_verified_fire(
            warm_moving_frame(10), previous_gray, classifier, min_area=250, confidence_threshold=0.90
        )
        self.assertEqual(candidate_count, 0)
        self.assertEqual(regions, [])
        self.assertEqual(classifier.calls, 0)

    def test_static_warm_object_is_not_a_motion_candidate(self) -> None:
        classifier = FakeClassifier(0.99)
        frame = warm_moving_frame(0)
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, _, regions, candidate_count = detect_verified_fire(
            frame, previous_gray, classifier, min_area=250, confidence_threshold=0.90
        )
        self.assertEqual(candidate_count, 0)
        self.assertEqual(regions, [])
    def test_unavailable_ai_model_optical_fallback_detects_fire(self) -> None:
        classifier = FakeClassifier(None, available=False)
        previous_gray = cv2.cvtColor(warm_moving_frame(0), cv2.COLOR_BGR2GRAY)
        _, _, regions, candidate_count = detect_verified_fire(
            warm_moving_frame(10), previous_gray, classifier, min_area=250, confidence_threshold=0.90,
            allow_heuristic_fallback=True,
        )
        self.assertGreater(candidate_count, 0)
        self.assertTrue(regions)
        self.assertEqual(classifier.calls, 0)


if __name__ == "__main__":
    unittest.main()
