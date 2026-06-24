import os
import sys
import unittest
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from invariant_processing import FaceRecognitionObservationAdapter, MediaReference


class FaceRecognitionObservationAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = FaceRecognitionObservationAdapter(model_version_fallback="test-model")
        self.observed_at = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)

    def test_allow_response_normalizes_to_observation(self) -> None:
        response = {
            "ok": True,
            "facesDetected": 1,
            "frDecision": "ALLOW",
            "model": "insightface(buffalo_l):SCRFD+ArcFace",
            "bestMatch": {"id": "tom", "score": 0.81},
            "faces": [
                {
                    "bbox": [10, 20, 110, 160],
                    "matchId": "tom",
                    "matchScore": 0.81,
                }
            ],
        }

        observations = self.adapter.to_observations(
            response,
            camera_id="cam-1",
            site_id="site-a",
            observed_at=self.observed_at,
            media_ref=MediaReference(uri="s3://bucket/frame-1.jpg", sha256="abc"),
            observation_id_prefix="obs",
        )

        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation.observation_id, "obs:0")
        self.assertEqual(observation.person_id, "tom")
        self.assertEqual(observation.camera_id, "cam-1")
        self.assertEqual(observation.site_id, "site-a")
        self.assertEqual(observation.decision, "ALLOW")
        self.assertEqual(observation.match_score, 0.81)
        self.assertEqual(observation.bbox.as_tuple(), (10.0, 20.0, 110.0, 160.0))
        self.assertEqual(observation.media_ref.uri, "s3://bucket/frame-1.jpg")
        self.assertEqual(observation.media_ref.sha256, "abc")
        self.assertEqual(observation.model_version, "insightface(buffalo_l):SCRFD+ArcFace")
        self.assertEqual(observation.metadata["facesDetected"], 1)
        self.assertEqual(observation.metadata["bestMatch"], {"id": "tom", "score": 0.81})

    def test_unknown_response_with_face_is_safe(self) -> None:
        response = {
            "ok": True,
            "facesDetected": 1,
            "frDecision": "UNKNOWN",
            "faces": [{"bbox": [0, 0, 20, 20], "matchId": None, "matchScore": 0.2}],
        }

        observations = self.adapter.to_observations(
            response,
            camera_id="cam-2",
            site_id="site-a",
            observed_at=self.observed_at,
            media_ref="s3://bucket/frame-2.jpg",
        )

        self.assertEqual(len(observations), 1)
        self.assertIsNone(observations[0].person_id)
        self.assertEqual(observations[0].decision, "UNKNOWN")
        self.assertEqual(observations[0].media_ref.uri, "s3://bucket/frame-2.jpg")
        self.assertEqual(observations[0].model_version, "test-model")

    def test_no_face_and_error_responses_return_empty_list(self) -> None:
        no_face = {"ok": True, "facesDetected": 0, "frDecision": "NO_FACE", "faces": []}
        error = {"ok": False, "facesDetected": 0, "frDecision": "ERROR", "error": "decode failed", "faces": []}

        self.assertEqual(
            self.adapter.to_observations(no_face, camera_id="cam-1", observed_at=self.observed_at),
            [],
        )
        self.assertEqual(
            self.adapter.to_observations(error, camera_id="cam-1", observed_at=self.observed_at),
            [],
        )


if __name__ == "__main__":
    unittest.main()
