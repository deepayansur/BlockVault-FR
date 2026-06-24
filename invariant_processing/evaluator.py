"""Invariant evaluator service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol

from invariant_processing.domain import EvaluationResult, Observation, as_utc_datetime
from invariant_processing.errors import InvariantValidationError
from invariant_processing.invariants import InvariantStatus, SecurityInvariant
from invariant_processing.registry import RegistrySnapshot


class ObservationProvider(Protocol):
    """Protocol for future stores that can provide observations to the evaluator."""

    def observations(self) -> Iterable[Observation]:
        """Return normalized observations."""


@dataclass(frozen=True)
class InvariantEvaluator:
    """Service that validates and evaluates security invariants."""

    registry: RegistrySnapshot

    def evaluate(
        self,
        invariant: SecurityInvariant,
        observations: Iterable[Observation],
        evaluation_time: datetime | str | None = None,
    ) -> EvaluationResult:
        evaluated_at = as_utc_datetime(evaluation_time or datetime.now(timezone.utc))
        try:
            invariant.validate(self.registry)
        except InvariantValidationError as exc:
            return EvaluationResult(
                invariant_id=invariant.invariant_id or "invalid",
                status=InvariantStatus.INVALID,
                evaluated_at=evaluated_at,
                reason=str(exc),
                validation_errors=exc.errors,
            )
        return invariant.evaluate(observations, self.registry, evaluated_at)

    def evaluate_from_provider(
        self,
        invariant: SecurityInvariant,
        provider: ObservationProvider,
        evaluation_time: datetime | str | None = None,
    ) -> EvaluationResult:
        return self.evaluate(invariant, provider.observations(), evaluation_time=evaluation_time)
