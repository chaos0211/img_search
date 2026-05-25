from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from backend.app.models.self_similarity_embedding import SelfSimilarityEmbedding
from backend.app.services.training_service import EmbeddingTrainingService


class SelfSimilarityEmbeddingTest(unittest.TestCase):
    def test_training_device_options_keep_cuda_visible_when_unavailable(self) -> None:
        with (
            patch("backend.app.services.training_service.torch.cuda.is_available", return_value=False),
            patch.object(EmbeddingTrainingService, "_mps_available", return_value=False),
        ):
            service = EmbeddingTrainingService()
            devices = service.available_devices()

        self.assertEqual([item["value"] for item in devices], ["auto", "cuda", "cpu"])
        cuda = next(item for item in devices if item["value"] == "cuda")
        self.assertFalse(cuda["available"])

    def test_training_device_resolves_cuda_index_when_available(self) -> None:
        with (
            patch("backend.app.services.training_service.torch.cuda.is_available", return_value=True),
            patch("backend.app.services.training_service.torch.cuda.device_count", return_value=2),
            patch("backend.app.services.training_service.torch.cuda.get_device_name", side_effect=["NVIDIA A", "NVIDIA B"]),
            patch.object(EmbeddingTrainingService, "_mps_available", return_value=False),
        ):
            service = EmbeddingTrainingService()
            devices = service.available_devices()
            resolved = service.resolve_device("cuda:1")

        self.assertIn("cuda:1", [item["value"] for item in devices])
        self.assertEqual(str(resolved), "cuda:1")

    def test_forward_uses_tensor_encoder_and_returns_normalized_embedding(self) -> None:
        torch.manual_seed(7)
        model = SelfSimilarityEmbedding(target_dim=16, channels=32, neighborhood_size=7)
        model.eval()
        feature_map = torch.randn(2, 32, 7, 7)

        with torch.inference_mode():
            embeddings = model(feature_map)

        self.assertEqual(embeddings.shape, (2, 16))
        self.assertEqual(len(model.tensor_encoder.blocks), 3)
        self.assertEqual(model.config()["architecture"], SelfSimilarityEmbedding.architecture_name)
        norms = torch.linalg.norm(embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_invalid_neighborhood_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SelfSimilarityEmbedding(target_dim=16, channels=32, neighborhood_size=6)

    def test_checkpoint_metadata_rejects_legacy_projection_embedder(self) -> None:
        service = EmbeddingTrainingService()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "legacy.pt"
            torch.save(
                {
                    "feature_dim": 16,
                    "embedder": {
                        "projection.0.weight": torch.randn(16, 64),
                        "projection.0.bias": torch.randn(16),
                    },
                },
                checkpoint_path,
            )
            with self.assertRaises(ValueError):
                service.embedding_checkpoint_metadata(str(checkpoint_path))

    def test_checkpoint_metadata_accepts_self_similarity_architecture(self) -> None:
        service = EmbeddingTrainingService()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "self_similarity.pt"
            torch.save(
                {
                    "feature_dim": 1024,
                    "architecture": SelfSimilarityEmbedding.architecture_name,
                    "embedder": SelfSimilarityEmbedding(target_dim=1024).state_dict(),
                },
                checkpoint_path,
            )
            metadata = service.embedding_checkpoint_metadata(str(checkpoint_path))
            self.assertEqual(metadata["architecture"], SelfSimilarityEmbedding.architecture_name)
            self.assertEqual(metadata["checkpoint_size_bytes"], str(checkpoint_path.stat().st_size))


if __name__ == "__main__":
    unittest.main()
