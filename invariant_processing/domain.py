"""Domain value objects for normalized observations and invariant evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Tuple

from invariant_processing.errors import DomainValidationError

if TYPE_CHECKING:
    from invariant_processing.invariants import InvariantStatus


_WINDOW_RE = re.compile(r"^\s*(\d+)\s*([mhdMHD])\s*$")
_WINDOW_MULTIPLIERS = {
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
}


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise DomainValidationError(f"{field_name} must be non-empty.")
    return normalized


def as_utc_datetime(value: datetime | str) -> datetime:
    """Normalize a datetime or ISO timestamp into a timezone-aware UTC datetime."""

    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise DomainValidationError(f"Invalid datetime value: {value!r}.") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise DomainValidationError(f"Expected datetime or ISO timestamp, got {type(value).__name__}.")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _freeze_metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return MappingProxyType(dict(metadata or {}))


@dataclass(frozen=True)
class BoundingBox:
    """Face bounding box in image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_sequence(cls, values: Iterable[Any]) -> "BoundingBox":
        coords = tuple(float(value) for value in values)
        if len(coords) != 4:
            raise DomainValidationError("BoundingBox requires exactly four coordinates.")
        return cls(*coords)

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise DomainValidationError("BoundingBox max coordinates must be greater than min coordinates.")

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class MediaReference:
    """Reference to image/video evidence stored outside the domain model."""

    uri: str
    sha256: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    storage_provider: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _require_non_empty(self.uri, "media uri"))
        if self.size_bytes is not None and self.size_bytes < 0:
            raise DomainValidationError("media size_bytes must be non-negative.")


@dataclass(frozen=True)
class Person:
    person_id: str
    display_name: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "person_id", _require_non_empty(self.person_id, "person_id"))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class Camera:
    camera_id: str
    site_id: Optional[str] = None
    name: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _require_non_empty(self.camera_id, "camera_id"))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class CameraSet:
    camera_set_id: str
    camera_ids: frozenset[str]
    site_id: Optional[str] = None
    name: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_set_id", _require_non_empty(self.camera_set_id, "camera_set_id"))
        normalized = frozenset(_require_non_empty(camera_id, "camera_id") for camera_id in self.camera_ids)
        object.__setattr__(self, "camera_ids", normalized)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def __len__(self) -> int:
        return len(self.camera_ids)

    def contains(self, camera_id: str) -> bool:
        return camera_id in self.camera_ids


@dataclass(frozen=True)
class EvaluationWindow:
    """Duration used to evaluate a temporal invariant."""

    seconds: int
    raw: Optional[str] = None

    @classmethod
    def parse(cls, value: str) -> "EvaluationWindow":
        match = _WINDOW_RE.match(value or "")
        if not match:
            raise DomainValidationError("Evaluation window must use m, h, or d, such as '30m', '2h', or '1d'.")
        amount = int(match.group(1))
        unit = match.group(2).lower()
        return cls(seconds=amount * _WINDOW_MULTIPLIERS[unit], raw=value.strip())

    def __post_init__(self) -> None:
        if int(self.seconds) <= 0:
            raise DomainValidationError("Evaluation window must be positive.")
        object.__setattr__(self, "seconds", int(self.seconds))

    def to_timedelta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    def start_for(self, evaluation_time: datetime | str) -> datetime:
        return as_utc_datetime(evaluation_time) - self.to_timedelta()


@dataclass(frozen=True)
class Observation:
    """Normalized face-recognition observation used by invariant evaluators."""

    observation_id: str
    camera_id: str
    observed_at: datetime | str
    decision: str
    person_id: Optional[str] = None
    site_id: Optional[str] = None
    match_score: Optional[float] = None
    bbox: Optional[BoundingBox] = None
    media_ref: Optional[MediaReference] = None
    model_version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _require_non_empty(self.observation_id, "observation_id"))
        object.__setattr__(self, "camera_id", _require_non_empty(self.camera_id, "camera_id"))
        object.__setattr__(self, "observed_at", as_utc_datetime(self.observed_at))
        object.__setattr__(self, "decision", _require_non_empty(self.decision, "decision"))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if self.person_id is not None:
            object.__setattr__(self, "person_id", _require_non_empty(self.person_id, "person_id"))
        if self.match_score is not None:
            score = float(self.match_score)
            if score < 0 or score > 1:
                raise DomainValidationError("match_score must be between 0 and 1.")
            object.__setattr__(self, "match_score", score)

    def matches_person(self, person_id: str, minimum_score: float) -> bool:
        return self.person_id == person_id and self.match_score is not None and self.match_score >= minimum_score


@dataclass(frozen=True)
class EvaluationResult:
    """Result returned by an invariant evaluation."""

    invariant_id: str
    status: "InvariantStatus"
    evaluated_at: datetime | str
    window_start: Optional[datetime | str] = None
    window_end: Optional[datetime | str] = None
    matching_camera_ids: frozenset[str] = field(default_factory=frozenset)
    required_camera_count: int = 0
    observed_camera_count: int = 0
    reason: str = ""
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "invariant_id", _require_non_empty(self.invariant_id, "invariant_id"))
        object.__setattr__(self, "evaluated_at", as_utc_datetime(self.evaluated_at))
        if self.window_start is not None:
            object.__setattr__(self, "window_start", as_utc_datetime(self.window_start))
        if self.window_end is not None:
            object.__setattr__(self, "window_end", as_utc_datetime(self.window_end))
        object.__setattr__(self, "matching_camera_ids", frozenset(self.matching_camera_ids))
        object.__setattr__(self, "validation_errors", tuple(self.validation_errors))
        object.__setattr__(self, "details", _freeze_metadata(self.details))
