"""Typed errors for invariant processing."""

from __future__ import annotations

from typing import Iterable, Tuple


class InvariantProcessingError(Exception):
    """Base error for invariant-processing failures."""


class DomainValidationError(InvariantProcessingError):
    """Raised when a domain value object is malformed."""


class InvariantValidationError(InvariantProcessingError):
    """Raised when an invariant is not valid for a registry snapshot."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors: Tuple[str, ...] = tuple(str(error) for error in errors)
        message = "; ".join(self.errors) if self.errors else "Invariant validation failed."
        super().__init__(message)


class InvariantEvaluationError(InvariantProcessingError):
    """Raised when invariant evaluation cannot complete."""
