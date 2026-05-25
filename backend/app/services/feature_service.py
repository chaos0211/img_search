from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from backend.app.config import settings
from backend.app.models.resnet_backbone import ResNet101Backbone
from backend.app.models.self_similarity_embedding import SelfSimilarityEmbedding
from backend.app.utils.file_utils import resolve_project_path
from backend.app.utils.metric_utils import normalize_vector


class FeatureService:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.baseline_backbone = ResNet101Backbone().to(self.device).eval()
        self.backbone = ResNet101Backbone().to(self.device).eval()
        self.embedder = SelfSimilarityEmbedding(target_dim=settings.feature_dim).to(self.device).eval()
        self.checkpoint_path = self._load_checkpoint()

    def extract(self, image_path: str, mode: str = "embedding") -> np.ndarray:
        if mode != "baseline" and self.checkpoint_path is None:
            raise ValueError("缺少有效自相似特征模型权重")
        tensor = self.backbone.preprocess(resolve_project_path(image_path)).to(self.device)
        with torch.inference_mode():
            if mode == "baseline":
                _feature_map, pooled = self.baseline_backbone.forward_features(tensor)
                vector = pooled.squeeze(0).cpu().numpy()
            else:
                feature_map, _pooled = self.backbone.forward_features(tensor)
                vector = self.embedder(feature_map).squeeze(0).cpu().numpy()
        return normalize_vector(vector.astype(np.float32))

    def _load_checkpoint(self) -> str | None:
        checkpoint_path = self._latest_checkpoint()
        if checkpoint_path is None:
            return None
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            architecture = self._checkpoint_architecture(checkpoint)
            if architecture != self.embedder.architecture_name:
                return None
            self.backbone.load_state_dict(checkpoint.get("backbone", {}), strict=False)
            self.embedder.load_state_dict(checkpoint.get("embedder", {}), strict=True)
        except Exception:
            return None
        self.backbone.eval()
        self.embedder.eval()
        return str(checkpoint_path)

    @staticmethod
    def _load_matching_state(module: torch.nn.Module, state_dict: dict) -> None:
        current = module.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        module.load_state_dict(compatible, strict=False)

    def _checkpoint_architecture(self, checkpoint: dict) -> str | None:
        architecture = checkpoint.get("architecture")
        if architecture:
            return str(architecture)
        keys = set(checkpoint.get("embedder", {}).keys())
        if any(key.startswith(("tensor_encoder.", "feature_fusion.", "input_projection.", "initial_norm.")) for key in keys):
            return self.embedder.architecture_name
        if any(key.startswith("projection.") for key in keys):
            return "legacy_projection_embedding"
        return None

    @staticmethod
    def _latest_checkpoint() -> Path | None:
        for name in ("embedding_best.pt", "embedding_latest.pt"):
            candidate = settings.project_root / "checkpoints" / name
            if candidate.exists():
                return candidate
        return None


@lru_cache(maxsize=1)
def get_feature_service() -> FeatureService:
    return FeatureService()
