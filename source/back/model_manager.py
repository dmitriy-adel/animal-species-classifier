import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WildlifeClassifier(nn.Module):
    """
    Архитектура модели: backbone (DINOv2) + embedding head + classifier head.
    Идентична той, что использовалась при обучении.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 256, backbone_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        backbone_out = self.backbone.config.hidden_size

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]  
        embedding = self.embedding_head(features)
        logits = self.classifier(embedding)
        return logits, embedding


class ModelManager:
    def __init__(self) -> None:
        self._path: str | None = None
        self._is_initialized: bool = False
        self._processor = None
        self._model: WildlifeClassifier | None = None
        self._device: torch.device | None = None
        self._class_to_idx: dict[str, int] | None = None
        self._idx_to_class: dict[int, str] | None = None

    def initialize(self, path: str, model_filename: str) -> None:
        """
        path — путь к директории, содержащей:
            - model.pt          (state_dict модели)
            - class_to_idx.json (маппинг "название вида" -> индекс класса)
        """
        self._path = path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classes_path = os.path.join(path, "class_to_idx.json")
        with open(classes_path, "r", encoding="utf-8") as f:
            self._class_to_idx = json.load(f)
            
        self._idx_to_class = {v: k for k, v in self._class_to_idx.items()}

        self._model = WildlifeClassifier(num_classes=len(self._class_to_idx))
        weights_path = os.path.join(path, model_filename)
        state_dict = torch.load(weights_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._processor = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self._is_initialized = True

    def classify(self, image: Image.Image) -> tuple[str, float]:
        self._ensure_initialized()

        tensor = self._preprocess(image)

        with torch.no_grad():
            logits, _ = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        label = self._idx_to_class[pred_idx.item()]
        return label, round(confidence.item(), 4)

    def vectorize(self, image: Image.Image) -> list[float]:
        self._ensure_initialized()

        tensor = self._preprocess(image)

        with torch.no_grad():
            _, embedding = self._model(tensor)

        return embedding.squeeze(0).cpu().tolist()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image_np = np.array(image)
        transformed = self._processor(image=image_np)["image"]
        tensor = transformed.unsqueeze(0).to(self._device)
        return tensor

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "Модель не инициализирована. Сначала вызовите initialize(path)."
            )