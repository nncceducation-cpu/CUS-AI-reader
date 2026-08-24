import json

from cus_ai.model import ModelManifest, discover_model


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

