"""Reusable invariant-processing domain layer for BlockVault V2."""

from invariant_processing.adapters import FaceRecognitionObservationAdapter
from invariant_processing.domain import (
    BoundingBox,
    Camera,
    CameraSet,
    EvaluationResult,
    EvaluationWindow,
    MediaReference,
    Observation,
    Person,
)
from invariant_processing.evaluator import InvariantEvaluator
from invariant_processing.invariants import InvariantStatus, PresenceCoverageInvariant, SecurityInvariant
from invariant_processing.registry import RegistrySnapshot

__all__ = [
    "BoundingBox",
    "Camera",
    "CameraSet",
    "EvaluationResult",
    "EvaluationWindow",
    "FaceRecognitionObservationAdapter",
    "InvariantEvaluator",
    "InvariantStatus",
    "MediaReference",
    "Observation",
    "Person",
    "PresenceCoverageInvariant",
    "RegistrySnapshot",
    "SecurityInvariant",
]
