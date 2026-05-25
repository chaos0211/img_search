from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.indexers.faiss_index import faiss
from backend.app.indexers.hnsw_index import HNSWIndex, hnswlib
from backend.app.indexers.pq_index import PQIndex
from backend.app.services.offline_evaluation_service import OfflineEvaluationService


class OfflineEvaluationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.service = OfflineEvaluationService()
        self.service.index_root = root / "indexes"
        self.service.index_root.mkdir(parents=True, exist_ok=True)
        self.service.metric_root = root / "metrics"
        self.service.metric_root.mkdir(parents=True, exist_ok=True)
        self.service.matrix_root = root / "matrix_cache"
        self.service.matrix_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _toy_gallery_query():
        gallery_vectors = np.asarray(
            [
                [1.00, 0.00, 0.00, 0.00],
                [0.96, 0.04, 0.00, 0.00],
                [0.00, 1.00, 0.00, 0.00],
                [0.03, 0.95, 0.02, 0.00],
                [0.00, 0.00, 1.00, 0.00],
                [0.00, 0.00, 0.95, 0.05],
            ],
            dtype=np.float64,
        )
        gallery_labels = ["airplane", "airplane", "dog", "dog", "ship", "ship"]
        gallery_ids = [f"gallery-{index}" for index in range(len(gallery_vectors))]
        query_vectors = np.asarray(
            [
                [1.00, 0.00, 0.00, 0.00],
                [0.00, 1.00, 0.00, 0.00],
            ],
            dtype=np.float64,
        )
        query_labels = ["airplane", "dog"]
        query_ids = ["query-airplane", "query-dog"]
        return gallery_vectors, gallery_labels, gallery_ids, query_vectors, query_labels, query_ids

    def _evaluate(self, index_type: str):
        gallery_vectors, gallery_labels, gallery_ids, query_vectors, query_labels, query_ids = self._toy_gallery_query()
        return self.service.evaluate_loaded(
            gallery_vectors=gallery_vectors,
            gallery_labels=gallery_labels,
            gallery_ids=gallery_ids,
            query_vectors=query_vectors,
            query_labels=query_labels,
            query_ids=query_ids,
            gallery_id="toy-gallery",
            query_set_id="toy-query",
            index_type=index_type,
            top_k=2,
            result_name=f"{index_type}.json",
            feature_scheme="baseline",
            feature_label="ResNet101",
            run_id=f"run-{index_type}",
            rerank=False,
        )

    def test_exact_indexes_use_same_normalized_similarity(self) -> None:
        brute = self._evaluate("brute")
        self.assertEqual(brute["galleryId"], "toy-gallery")
        self.assertEqual(brute["querySetId"], "toy-query")
        self.assertAlmostEqual(brute["mapAtK"], 1.0)
        self.assertAlmostEqual(brute["recallAtK"], 1.0)
        self.assertAlmostEqual(brute["precisionAtK"], 1.0)
        self.assertGreaterEqual(brute["timingMs"]["indexSearchMs"], 0)
        self.assertGreater(brute["storageSize"]["featureSizeBytes"], 0)
        self.assertGreater(brute["storageSize"]["queryFeatureSizeBytes"], 0)
        self.assertGreaterEqual(brute["storageSize"]["totalStorageSizeBytes"], brute["storageSize"]["featureSizeBytes"])

        if faiss is None:
            self.skipTest("faiss is not installed")
        flat = self._evaluate("faiss")
        self.assertEqual(flat["indexLibrary"], "faiss")
        self.assertEqual(flat["indexMethod"], "FlatIP")
        self.assertEqual(flat["indexMetadata"]["index_class"], "IndexFlatIP")
        self.assertEqual(flat["indexMetadata"]["metric"], "inner_product")
        self.assertEqual(flat["mapAtK"], brute["mapAtK"])
        self.assertEqual(flat["recallAtK"], brute["recallAtK"])
        self.assertEqual(flat["precisionAtK"], brute["precisionAtK"])

    def test_query_id_is_removed_when_query_appears_in_gallery(self) -> None:
        result = self.service.evaluate_loaded(
            gallery_vectors=np.asarray([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]], dtype=np.float64),
            gallery_labels=["cat", "cat", "dog"],
            gallery_ids=["shared-image", "cat-copy", "dog-1"],
            query_vectors=np.asarray([[1.0, 0.0]], dtype=np.float64),
            query_labels=["cat"],
            query_ids=["shared-image"],
            index_type="brute",
            top_k=1,
            result_name="self-match.json",
        )
        self.assertAlmostEqual(result["mapAtK"], 1.0)
        self.assertAlmostEqual(result["recallAtK"], 1.0)
        self.assertAlmostEqual(result["precisionAtK"], 1.0)

    def test_feature_manifest_loading_uses_float32_l2_normalized_cache(self) -> None:
        root = Path(self.tmp.name)
        feature_a = root / "a.npy"
        feature_b = root / "b.npy"
        np.save(feature_a, np.asarray([3.0, 4.0, 0.0], dtype=np.float64))
        np.save(feature_b, np.asarray([0.0, 0.0, 2.0], dtype=np.float64))
        manifest = root / "features.csv"
        pd.DataFrame(
            {
                "image_id": ["a", "b"],
                "label_name": ["cat", "dog"],
                "feature_path": [str(feature_a), str(feature_b)],
                "feature_mode": ["baseline", "baseline"],
                "checkpoint_path": ["", ""],
                "model_name": ["ResNet101", "ResNet101"],
            }
        ).to_csv(manifest, index=False, encoding="utf-8-sig")

        first = self.service.load_feature_dataset_payload(manifest)
        second = self.service.load_feature_dataset_payload(manifest)
        self.assertEqual(first["vectors"].dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(first["vectors"], axis=1), 1.0))
        self.assertTrue(np.array_equal(first["vectors"], second["vectors"]))
        self.assertEqual(first["ids"], ["a", "b"])
        self.assertTrue(first["datasetId"].startswith("features_"))

    def test_pq_train_add_flow_is_separate_and_persistable(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.9, 0.1],
            ],
            dtype=np.float32,
        )
        index = PQIndex(subvectors=2, cluster_count=2)
        with self.assertRaises(ValueError):
            index.add(vectors)
        index.train(vectors)
        self.assertTrue(index.is_trained)
        self.assertIsNone(index.codes)
        index.add(vectors)
        self.assertIsNotNone(index.codes)
        indices, scores = index.search(vectors[0], top_k=2)
        self.assertEqual(indices.shape, (2,))
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(index.metadata()["trained"])
        self.assertTrue(index.metadata()["added"])

        index_path = Path(self.tmp.name) / "pq.npz"
        index.save(index_path)
        loaded = PQIndex()
        loaded.load(index_path)
        self.assertTrue(loaded.metadata()["trained"])
        self.assertTrue(loaded.metadata()["added"])

    def test_hnsw_metadata_saves_runtime_parameters(self) -> None:
        if hnswlib is None:
            self.skipTest("hnswlib is not installed")
        index = HNSWIndex()
        index.build(np.eye(6, dtype=np.float32))
        index_path = Path(self.tmp.name) / "hnsw.bin"
        index.save(index_path)
        metadata = json.loads(index_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["index_method"], "HNSW")
        self.assertGreater(metadata["M"], 0)
        self.assertGreater(metadata["ef_construction"], 0)
        self.assertGreater(metadata["ef_search"], 0)


if __name__ == "__main__":
    unittest.main()
