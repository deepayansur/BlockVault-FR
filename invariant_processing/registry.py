"""Registry snapshots used to validate invariants without a database dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

from invariant_processing.domain import Camera, CameraSet, Person


@dataclass(frozen=True)
class RegistrySnapshot:
    """Immutable snapshot of known people, cameras, and camera sets."""

    people: Mapping[str, Person] = field(default_factory=dict)
    cameras: Mapping[str, Camera] = field(default_factory=dict)
    camera_sets: Mapping[str, CameraSet] = field(default_factory=dict)

    @classmethod
    def from_values(
        cls,
        people: Iterable[Person] = (),
        cameras: Iterable[Camera] = (),
        camera_sets: Iterable[CameraSet] = (),
    ) -> "RegistrySnapshot":
        return cls(
            people={person.person_id: person for person in people},
            cameras={camera.camera_id: camera for camera in cameras},
            camera_sets={camera_set.camera_set_id: camera_set for camera_set in camera_sets},
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "people", MappingProxyType(dict(self.people)))
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))
        object.__setattr__(self, "camera_sets", MappingProxyType(dict(self.camera_sets)))

    def has_person(self, person_id: str) -> bool:
        return person_id in self.people

    def has_camera(self, camera_id: str) -> bool:
        return camera_id in self.cameras

    def get_camera_set(self, camera_set_id: str) -> Optional[CameraSet]:
        return self.camera_sets.get(camera_set_id)

    def missing_camera_ids(self, camera_set: CameraSet) -> frozenset[str]:
        return frozenset(camera_id for camera_id in camera_set.camera_ids if camera_id not in self.cameras)
