"""Read only the four inspected historical reconstructions; never unpickle."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_verified_cache(directory, dataset, method, expected_shape):
    arm = {"q2": "qep_q2", "q15": "qep_q1.5"}[method]
    name = f"{dataset}_{dataset}_figure_1_{arm}.npz"
    manifest = json.loads((Path(__file__).parent / "historical_cache_manifest.json").read_text())
    expected = manifest["entries"][name]
    path = Path(directory) / name
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_hash != expected["file_sha256"]:
        raise ValueError(f"Historical cache file checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as cache:
        source = cache["predmean"]
        run_id = str(cache["run_id"])
        if str(source.dtype) != expected["dtype"] or run_id != expected["run_id"]:
            raise ValueError(f"Historical cache provenance mismatch: {path}")
        arr = np.ascontiguousarray(source, dtype="<f8")
    if tuple(arr.shape) != tuple(expected_shape) or list(arr.shape) != expected["shape"]:
        raise ValueError(f"Historical cache geometry mismatch: {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Nonfinite historical reconstruction: {path}")
    array_hash = hashlib.sha256(arr.tobytes()).hexdigest()
    if array_hash != expected["array_sha256"]:
        raise ValueError(f"Historical cache array checksum mismatch: {path}")
    return arr, dict(source="verified historical reconstruction; no new fitting",
        source_path=str(path.resolve()), file_sha256=file_hash, array_sha256=array_hash,
        run_id=run_id, historical_pipeline_runtime_s=expected["historical_pipeline_runtime_s"],
        runtime_note="Historical runtime includes the old full pipeline, not reconstruction alone",
        recorded_hyperparameter_summaries=expected["hypers"])
