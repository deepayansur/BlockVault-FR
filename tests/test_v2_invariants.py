import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from invariant_processing import (
    Camera,
    CameraSet,
    EvaluationWindow,
    InvariantEvaluator,
    InvariantStatus,
    Observation,
    Person,
    PresenceCoverageInvariant,
    RegistrySnapshot,
)
from invariant_processing.errors import DomainValidationError


class PresenceCoverageInvariantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        self.registry = RegistrySnapshot.from_values(
            people=[Person("tom"), Person("jane")],
            cameras=[Camera(f"cam-{idx}", site_id="warehouse") for idx in range(1, 11)],
            camera_sets=[CameraSet("warehouse-all", frozenset(f"cam-{idx}" for idx in range(1, 11)))],
        )
        self.invariant = PresenceCoverageInvariant(
            invariant_id="tom-coverage-2h",
            person_id="tom",
            camera_set_id="warehouse-all",
            window=EvaluationWindow.parse("2h"),
            min_distinct_cameras=6,
            match_threshold=0.65,
        )
        self.evaluator = InvariantEvaluator(self.registry)

    def observation(
        self,
        camera_id: str,
        person_id: str = "tom",
        score: float = 0.8,
        observed_at: datetime | None = None,
    ) -> Observation:
        return Observation(
            observation_id=f"{camera_id}-{person_id}-{score}-{observed_at or self.now}",
            camera_id=camera_id,
            site_id="warehouse",
            observed_at=observed_at or self.now,
            decision="ALLOW",
            person_id=person_id,
            match_score=score,
            model_version="test-model",
        )

    def test_valid_invariant_passes_validation(self) -> None:
        self.assertEqual(self.invariant.validation_errors(self.registry), [])

    def test_unknown_person_fails_validation(self) -> None:
        invariant = PresenceCoverageInvariant(
            invariant_id="bad-person",
            person_id="unknown",
            camera_set_id="warehouse-all",
            window=EvaluationWindow.parse("2h"),
            min_distinct_cameras=1,
            match_threshold=0.65,
        )

        result = self.evaluator.evaluate(invariant, [], evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.INVALID)
        self.assertIn("Unknown person_id", result.reason)

    def test_unknown_camera_set_fails_validation(self) -> None:
        invariant = PresenceCoverageInvariant(
            invariant_id="bad-camera-set",
            person_id="tom",
            camera_set_id="missing",
            window=EvaluationWindow.parse("2h"),
            min_distinct_cameras=1,
            match_threshold=0.65,
        )

        result = self.evaluator.evaluate(invariant, [], evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.INVALID)
        self.assertIn("Unknown camera_set_id", result.reason)

    def test_invalid_threshold_fails_validation(self) -> None:
        invariant = PresenceCoverageInvariant(
            invariant_id="bad-threshold",
            person_id="tom",
            camera_set_id="warehouse-all",
            window=EvaluationWindow.parse("2h"),
            min_distinct_cameras=1,
            match_threshold=1.2,
        )

        result = self.evaluator.evaluate(invariant, [], evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.INVALID)
        self.assertIn("match_threshold", result.reason)

    def test_invalid_window_fails_validation(self) -> None:
        with self.assertRaises(DomainValidationError):
            EvaluationWindow.parse("2w")

    def test_min_distinct_cameras_cannot_exceed_camera_set_size(self) -> None:
        invariant = PresenceCoverageInvariant(
            invariant_id="too-many-cameras",
            person_id="tom",
            camera_set_id="warehouse-all",
            window=EvaluationWindow.parse("2h"),
            min_distinct_cameras=11,
            match_threshold=0.65,
        )

        result = self.evaluator.evaluate(invariant, [], evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.INVALID)
        self.assertIn("min_distinct_cameras", result.reason)

    def test_satisfied_when_person_appears_in_required_distinct_cameras(self) -> None:
        observations = [self.observation(f"cam-{idx}") for idx in range(1, 7)]

        result = self.evaluator.evaluate(self.invariant, observations, evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.SATISFIED)
        self.assertEqual(result.observed_camera_count, 6)
        self.assertEqual(result.matching_camera_ids, frozenset(f"cam-{idx}" for idx in range(1, 7)))

    def test_violated_when_person_appears_in_too_few_cameras(self) -> None:
        observations = [self.observation(f"cam-{idx}") for idx in range(1, 5)]

        result = self.evaluator.evaluate(self.invariant, observations, evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.VIOLATED)
        self.assertEqual(result.observed_camera_count, 4)

    def test_duplicate_observations_from_same_camera_count_once(self) -> None:
        observations = [
            self.observation("cam-1"),
            self.observation("cam-1", observed_at=self.now - timedelta(minutes=10)),
            self.observation("cam-2"),
            self.observation("cam-3"),
            self.observation("cam-4"),
            self.observation("cam-5"),
        ]

        result = self.evaluator.evaluate(self.invariant, observations, evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.VIOLATED)
        self.assertEqual(result.observed_camera_count, 5)

    def test_observations_outside_window_are_ignored(self) -> None:
        observations = [self.observation(f"cam-{idx}") for idx in range(1, 6)]
        observations.append(self.observation("cam-6", observed_at=self.now - timedelta(hours=3)))

        result = self.evaluator.evaluate(self.invariant, observations, evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.VIOLATED)
        self.assertEqual(result.observed_camera_count, 5)

    def test_low_score_observations_are_ignored(self) -> None:
        observations = [self.observation(f"cam-{idx}") for idx in range(1, 6)]
        observations.append(self.observation("cam-6", score=0.5))

        result = self.evaluator.evaluate(self.invariant, observations, evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.VIOLATED)
        self.assertEqual(result.observed_camera_count, 5)

    def test_wrong_person_observations_are_ignored(self) -> None:
        observations = [self.observation(f"cam-{idx}") for idx in range(1, 6)]
        observations.append(self.observation("cam-6", person_id="jane"))

        result = self.evaluator.evaluate(self.invariant, observations, evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.VIOLATED)
        self.assertEqual(result.observed_camera_count, 5)

    def test_no_matching_observations_returns_no_data(self) -> None:
        observations = [
            self.observation("cam-1", person_id="jane"),
            self.observation("cam-2", score=0.1),
            self.observation("cam-3", observed_at=self.now - timedelta(hours=3)),
        ]

        result = self.evaluator.evaluate(self.invariant, observations, evaluation_time=self.now)

        self.assertEqual(result.status, InvariantStatus.NO_DATA)
        self.assertEqual(result.observed_camera_count, 0)


if __name__ == "__main__":
    unittest.main()
