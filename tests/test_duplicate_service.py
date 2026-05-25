from __future__ import annotations

import unittest

import numpy as np

from backend.app.services.duplicate_service import DuplicateService


class _StubGalleryService:
    def load_feature_rows(self, include_thumbnails: bool = False):
        return [
            {"id": 1, "originalName": "a.png", "filePath": "a.png", "feature": np.asarray([1.0, 0.0], dtype=np.float32), "featureModel": "toy", "featureDim": 2},
            {"id": 2, "originalName": "a_copy.png", "filePath": "a_copy.png", "feature": np.asarray([0.999, 0.001], dtype=np.float32), "featureModel": "toy", "featureDim": 2},
            {"id": 3, "originalName": "b.png", "filePath": "b.png", "feature": np.asarray([0.0, 1.0], dtype=np.float32), "featureModel": "toy", "featureDim": 2},
            {"id": 4, "originalName": "b_copy.png", "filePath": "b_copy.png", "feature": np.asarray([0.001, 0.999], dtype=np.float32), "featureModel": "toy", "featureDim": 2},
        ]


class DuplicateServiceTest(unittest.TestCase):
    def test_threshold_evaluation_uses_gallery_features(self) -> None:
        service = DuplicateService()
        service.gallery_service = _StubGalleryService()
        service._build_validation_pairs = lambda _items, _vectors, _sample_size: [
            {"similarity": 0.96, "isDuplicate": True},
            {"similarity": 0.98, "isDuplicate": True},
            {"similarity": 0.94, "isDuplicate": False},
            {"similarity": 0.97, "isDuplicate": False},
        ]

        result = service.evaluate_thresholds(thresholds=[0.95, 0.97], sample_size=2)

        self.assertEqual(result["galleryCount"], 4)
        self.assertEqual(result["featureModel"], "toy")
        self.assertEqual(result["featureDim"], 2)
        self.assertEqual(result["sampleSize"], 2)
        self.assertEqual(result["positivePairCount"], 2)
        self.assertEqual(result["negativePairCount"], 2)
        self.assertEqual(result["recommendedThreshold"], 0.95)
        self.assertEqual(result["rows"][0]["tp"], 2)
        self.assertEqual(result["rows"][0]["fp"], 1)
        self.assertEqual(result["rows"][0]["fn"], 0)
        self.assertEqual(result["rows"][0]["tn"], 1)
        self.assertAlmostEqual(result["rows"][0]["precision"], 0.6667)
        self.assertAlmostEqual(result["rows"][0]["recall"], 1.0)
        self.assertAlmostEqual(result["rows"][0]["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
