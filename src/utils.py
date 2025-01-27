from datetime import datetime
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)


def convert_to_serialisable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serialisable(e) for e in obj]
    return obj


def save_json_results(data, output_dir, filename_prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{filename_prefix}_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(convert_to_serialisable(data), f, indent=2)
    return filepath


def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    return metrics


class CacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_stats(self, file_path):
        path = Path(file_path)
        if not path.exists():
            return {}
        stats = path.stat()
        return {"size": stats.st_size, "mtime": stats.st_mtime, "name": path.name}

    def get_cache_key(self, data_path, config):
        metadata = {"file_stats": self._get_file_stats(data_path), "config": config}
        return hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def save_data(self, data, cache_key):
        cache_path = self.cache_dir / cache_key
        joblib.dump(data, cache_path)

    def load_data(self, cache_key):
        cache_path = self.cache_dir / cache_key
        return joblib.load(cache_path) if cache_path.exists() else None
