# V2 Status Update

Date: 2026-05-08

## Current Status
V2 is in early foundation work inside `BlockVault-FR`. The first completed slice is the new invariant-processing domain layer, added without changing the existing face-recognition runtime.

## Completed
- Added a standalone `invariant_processing/` package for V2 logic.
- Introduced reusable domain objects for observations, cameras, camera sets, media references, evaluation windows, and evaluation results.
- Added registry snapshots for validating invariants without needing a database yet.
- Implemented the first production-style invariant: `PresenceCoverageInvariant`.
- Added an evaluator service and an adapter for converting current FR `/verify` outputs into normalized observations.
- Added focused unit tests for FR response normalization and invariant validation/evaluation behavior.

## Not Started Yet
- Postgres-backed metadata and observation storage
- Redpanda/Kafka event backbone
- S3-compatible media storage integration
- Monitoring agents, alerting, and background processing
- LLM-driven invariant authoring, querying, and benchmarking

## Next Recommended Step
Build the runtime around this new domain layer:
- persist observations and invariant definitions
- evaluate invariants on a schedule
- introduce alert state transitions

## Notes
- Existing `face_service` behavior remains unchanged in this phase.
- Full test discovery still has a pre-existing NumPy compatibility issue in the legacy eval metrics path; the new Phase 1 invariant tests pass.
