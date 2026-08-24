import json

import numpy as np
import pytest
from PIL import Image

from cus_ai.media import MediaFrame
from cus_ai.model import ModelManifest, OnnxFeatureModel, discover_model


def test_no_manifest_disables_model(tmp_path):
    manifest, warnings = discover_model(tmp_path)
    assert manifest is None
    assert warnings


def test_manifest_load_and_missing_weights_guard(tmp_path):
    payload = {
        "model_id": "test-model",
        "version": "1.2.3",
        "onnx_file": "missing.onnx",
        "input_size": [256, 256],
        "labels": ["a", "b"],
        "validated": False,
    }
    path = tmp_path / "test.manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest, warnings = discover_model(tmp_path)
    assert isinstance(manifest, ModelManifest)
    assert manifest.input_size == (256, 256)
    assert any("missing" in item.lower() for item in warnings)
    assert any("unvalidated" in item.lower() for item in warnings)


class FakeSession:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.float32)
        self.offset = 0

    def run(self, _outputs, inputs):
        batch_size = next(iter(inputs.values())).shape[0]
        result = self.rows[self.offset : self.offset + batch_size]
        self.offset += batch_size
        return [result]


def test_model_processes_every_frame_and_aggregates_by_plane():
    labels = ["plane_coronal", "plane_sagittal", "plane_other", "left_intraventricular_blood"]
    manifest = ModelManifest(
        model_id="fake",
        version="1",
        onnx_file="fake.onnx",
        input_size=(32, 32),
        labels=labels,
        validated=True,
        batch_size=2,
        plane_confidence_threshold=0.70,
        plane_margin=0.10,
    )
    model = object.__new__(OnnxFeatureModel)
    model.manifest = manifest
    model.input_name = "input"
    model.session = FakeSession(
        [
            [0.90, 0.05, 0.05, 0.70],
            [0.05, 0.90, 0.05, 0.80],
            [0.51, 0.49, 0.00, 0.99],
        ]
    )
    frames = [
        MediaFrame("clip.avi", index, Image.new("RGB", (64, 64)), "video")
        for index in range(3)
    ]

    prediction = model.predict(frames)
    assert prediction.processed_frame_count == 3
    assert [row.frame_index for row in prediction.frame_predictions] == [0, 1, 2]
    assert prediction.plane_counts["coronal"] == 1
    assert prediction.plane_counts["sagittal"] == 1
    assert prediction.plane_counts["indeterminate"] == 1
    assert prediction.probabilities["left_intraventricular_blood"] == pytest.approx(0.7)
