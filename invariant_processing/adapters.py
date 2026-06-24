"""Adapters from current face-recognition outputs to normalized observations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional

from invariant_processing.domain import BoundingBox, MediaReference, Observation, as_utc_datetime
from invariant_processing.errors import DomainValidationError


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_media_ref(media_ref: MediaReference | str | None) -> Optional[MediaReference]:
    if media_ref is None or isinstance(media_ref, MediaReference):
        return media_ref
    return MediaReference(uri=str(media_ref))


@dataclass(frozen=True)
class FaceRecognitionObservationAdapter:
    """Convert BlockVault-FR `/verify` responses into normalized observations."""

    model_version_fallback: Optional[str] = None

    def to_observations(
        self,
        verify_response: Mapping[str, Any],
        camera_id: str,
        site_id: Optional[str] = None,
        observed_at: datetime | str | None = None,
        media_ref: MediaReference | str | None = None,
        observation_id_prefix: Optional[str] = None,
    ) -> List[Observation]:
        timestamp = as_utc_datetime(observed_at or datetime.now(timezone.utc))
        faces = verify_response.get("faces")
        if not isinstance(faces, list) or not faces:
            return []

        decision = str(verify_response.get("frDecision") or "UNKNOWN")
        model_version = str(verify_response.get("model") or self.model_version_fallback or "")
        normalized_media = _coerce_media_ref(media_ref)
        prefix = observation_id_prefix or f"{camera_id}:{timestamp.isoformat()}"

        observations: List[Observation] = []
        for index, face in enumerate(faces):
            if not isinstance(face, Mapping):
                continue
            bbox = self._parse_bbox(face.get("bbox"))
            person_id = face.get("matchId")
            score = _optional_float(face.get("matchScore"))
            observations.append(
                Observation(
                    observation_id=f"{prefix}:{index}",
                    camera_id=camera_id,
                    site_id=site_id,
                    observed_at=timestamp,
                    decision=decision,
                    person_id=str(person_id) if person_id is not None else None,
                    match_score=score,
                    bbox=bbox,
                    media_ref=normalized_media,
                    model_version=model_version or None,
                    metadata={
                        "source": "blockvault-fr",
                        "facesDetected": verify_response.get("facesDetected"),
                        "bestMatch": verify_response.get("bestMatch"),
                    },
                )
            )
        return observations

    def _parse_bbox(self, values: Any) -> Optional[BoundingBox]:
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return None
        try:
            return BoundingBox.from_sequence(values)
        except (DomainValidationError, TypeError, ValueError):
            return None
