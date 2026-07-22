import os

import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset, load_from_disk

DATA_DIR = "data/"
MANIFEST_FILENAME = "cropped_filtered-final_manifest-v2-with_net_v_klass.csv"
DATASET_NAME = "dmitriy-zador/wildlife-animals"
DATASET_DIR_NAME = "wildlife-animals"
EMBEDDINGS_FILENAME = "records_embeddings.npy"
LABELS_FILENAME = "records_labels.npy"

SPLIT_NAME_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


class RecordsBase:
    def __init__(self) -> None:
        self._is_initialized: bool = False
        self._images: list[Image.Image] = []
        self._labels: list[str] = []
        self._embeddings: np.ndarray | None = None  

    def initialize(self, data_dir: str = DATA_DIR, model_manager=None) -> None:
        os.makedirs(data_dir, exist_ok=True)

        manifest_path = os.path.join(data_dir, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest {MANIFEST_FILENAME} wasnt found in {data_dir}"
            )
        df = pd.read_csv(manifest_path)

        dataset_dir = os.path.join(data_dir, DATASET_DIR_NAME)
        if not self._dataset_exists_locally(dataset_dir):
            self._download_and_cache_dataset(dataset_dir)

        hf_dataset = load_from_disk(dataset_dir)

        self._images, self._labels = self._align_dataframe_with_dataset(df, hf_dataset)

        embeddings_path = os.path.join(data_dir, EMBEDDINGS_FILENAME)
        labels_cache_path = os.path.join(data_dir, LABELS_FILENAME)

        if os.path.exists(embeddings_path) and os.path.exists(labels_cache_path):
            self._embeddings = np.load(embeddings_path)
            cached_labels = np.load(labels_cache_path, allow_pickle=True).tolist()
            if cached_labels != self._labels:
                self._embeddings = self._compute_embeddings(model_manager)
                np.save(embeddings_path, self._embeddings)
                np.save(labels_cache_path, np.array(self._labels, dtype=object))
                
        else:
            if model_manager is None:
                raise RuntimeError(
                    "cant calculate embeddings"
                )
            self._embeddings = self._compute_embeddings(model_manager)
            np.save(embeddings_path, self._embeddings)
            np.save(labels_cache_path, np.array(self._labels, dtype=object))

        self._is_initialized = True

    def find_nearest(self, query_vector: list[float], top_k: int = 1) -> tuple[Image.Image, float]:
        self._ensure_initialized()

        if self._embeddings is None or len(self._embeddings) == 0:
            raise LookupError("there is no images in db")

        query = np.array(query_vector, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        base_norms = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        )

        similarities = base_norms @ query_norm  

        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])

        return self._images[best_idx], round(best_similarity, 4)

    def _dataset_exists_locally(self, dataset_dir: str) -> bool:
        return os.path.isdir(dataset_dir) and os.path.exists(
            os.path.join(dataset_dir, "dataset_dict.json")
        )

    def _download_and_cache_dataset(self, dataset_dir: str) -> None:
        dataset = load_dataset(DATASET_NAME)  
        dataset.save_to_disk(dataset_dir)

    def _align_dataframe_with_dataset(self, df: pd.DataFrame, hf_dataset) -> tuple[list[Image.Image], list[str]]:
        images: list[Image.Image] = []
        labels: list[str] = []

        for manifest_split, hf_split in SPLIT_NAME_MAP.items():
            split_df = df[df["split"] == manifest_split].reset_index(drop=True)
            if len(split_df) == 0:
                continue

            hf_split_data = hf_dataset[hf_split]

            for i in range(len(split_df)):
                image = hf_split_data[i]["image"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                images.append(image.convert("RGB"))
                labels.append(split_df.loc[i, "name"])

        return images, labels

    def _compute_embeddings(self, model_manager) -> np.ndarray:
        vectors = []
        for image in self._images:
            vector = model_manager.vectorize(image)
            vectors.append(vector)
        return np.array(vectors, dtype=np.float32)

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "db wasnt initialized"
            )