"""Security invariant definitions and deterministic evaluation logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, List

from invariant_processing.domain import EvaluationResult, EvaluationWindow, Observation, as_utc_datetime
from invariant_processing.errors import InvariantValidationError
from invariant_processing.registry import RegistrySnapshot


class InvariantStatus(Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    NO_DATA = "NO_DATA"
    INVALID = "INVALID"


@dataclass(frozen=True)
class SecurityInvariant(ABC):
    """Base class for reusable security invariants."""

    invariant_id: str

    def validate(self, registry: RegistrySnapshot) -> None:
        errors = self.validation_errors(registry)
        if errors:
            raise InvariantValidationError(errors)

    @abstractmethod
    def validation_errors(self, registry: RegistrySnapshot) -> List[str]:
        """Return human-readable validation errors for this invariant."""

    @abstractmethod
    def evaluate(
        self,
        observations: Iterable[Observation],
        registry: RegistrySnapshot,
        evaluation_time: datetime,
    ) -> EvaluationResult:
        """Evaluate this invariant against normalized observations."""


@dataclass(frozen=True)
class PresenceCoverageInvariant(SecurityInvariant):
    """Require a person to be observed by a minimum number of distinct cameras."""

    person_id: str
    camera_set_id: str
    window: EvaluationWindow
    min_distinct_cameras: int
    match_threshold: float

    def validation_errors(self, registry: RegistrySnapshot) -> List[str]:
        errors: List[str] = []
        if not str(self.invariant_id or "").strip():
            errors.append("invariant_id must be non-empty.")
        if not str(self.person_id or "").strip():
            errors.append("person_id must be non-empty.")
        elif not registry.has_person(self.person_id):
            errors.append(f"Unknown person_id: {self.person_id}.")
        if not str(self.camera_set_id or "").strip():
            errors.append("camera_set_id must be non-empty.")

        camera_set = registry.get_camera_set(self.camera_set_id)
        if camera_set is None:
            errors.append(f"Unknown camera_set_id: {self.camera_set_id}.")
        elif len(camera_set) == 0:
            errors.append(f"Camera set {self.camera_set_id} must contain at least one camera.")
        else:
            missing_cameras = registry.missing_camera_ids(camera_set)
            if missing_cameras:
                errors.append(f"Camera set {self.camera_set_id} references unknown cameras: {sorted(missing_cameras)}.")

        if not isinstance(self.window, EvaluationWindow):
            errors.append("window must be an EvaluationWindow.")
        elif self.window.seconds <= 0:
            errors.append("window must be positive.")

        if self.min_distinct_cameras < 1:
            errors.append("min_distinct_cameras must be at least 1.")
        elif camera_set is not None and len(camera_set) > 0 and self.min_distinct_cameras > len(camera_set):
            errors.append("min_distinct_cameras cannot exceed the camera set size.")

        if self.match_threshold < 0 or self.match_threshold > 1:
            errors.append("match_threshold must be between 0 and 1.")
        return errors

    def evaluate(
        self,
        observations: Iterable[Observation],
        registry: RegistrySnapshot,
        evaluation_time: datetime,
    ) -> EvaluationResult:
        evaluated_at = as_utc_datetime(evaluation_time)
        window_start = self.window.start_for(evaluated_at)
        camera_set = registry.get_camera_set(self.camera_set_id)
        if camera_set is None:
            raise InvariantValidationError([f"Unknown camera_set_id: {self.camera_set_id}."])

        matching_camera_ids = set()
        considered = 0
        for observation in observations:
            if observation.camera_id not in camera_set.camera_ids:
                continue
            if observation.observed_at < window_start or observation.observed_at > evaluated_at:
                continue
            if not observation.matches_person(self.person_id, self.match_threshold):
                continue
            considered += 1
            matching_camera_ids.add(observation.camera_id)

        observed_count = len(matching_camera_ids)
        if observed_count >= self.min_distinct_cameras:
            status = InvariantStatus.SATISFIED
            reason = f"Observed {self.person_id} in {observed_count} distinct cameras."
        elif considered == 0:
            status = InvariantStatus.NO_DATA
            reason = f"No qualifying observations for {self.person_id} in the evaluation window."
        else:
            status = InvariantStatus.VIOLATED
            reason = (
                f"Observed {self.person_id} in {observed_count} distinct cameras; "
                f"requires {self.min_distinct_cameras}."
            )

        return EvaluationResult(
            invariant_id=self.invariant_id,
            status=status,
            evaluated_at=evaluated_at,
            window_start=window_start,
            window_end=evaluated_at,
            matching_camera_ids=frozenset(matching_camera_ids),
            required_camera_count=self.min_distinct_cameras,
            observed_camera_count=observed_count,
            reason=reason,
            details={
                "camera_set_id": self.camera_set_id,
                "person_id": self.person_id,
                "match_threshold": self.match_threshold,
                "qualifying_observations": considered,
            },
        )
