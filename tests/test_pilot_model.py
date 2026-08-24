from pathlib import Path

import numpy as np
from PIL import Image

from cus_ai.ai_consensus import grade_prediction
from cus_ai.media import MediaFrame
from cus_ai.model import PilotSimilarityModel, discover_model, load_model, prediction_to_json


ROOT = Path(__file__).resolve().parents[1]


def synthetic_frames(count: int = 3) -> list[MediaFrame]:
    return [
        MediaFrame(
            source_name="synthetic.png",
            frame_index=index,
            image=Image.fromarray(np.full((256, 256), 30 + index, dtype=np.uint8)),
            media_type="image",
        )
        for index in range(count)
    ]


def test_bundled_pilot_runs_every_supplied_frame_and_reaches_consensus_engine():
    manifest, warnings = discover_model(ROOT / "models")
    assert manifest is not None
    assert manifest.model_type == "pilot_similarity"
    assert manifest.validated is False
    assert any("unvalidated" in warning for warning in warnings)
    assert not any("failed SHA256" in warning for warning in warnings)
    model = load_model(ROOT / "models", manifest)
    assert isinstance(model, PilotSimilarityModel)
    prediction = prediction_to_json(model.predict(synthetic_frames()))
    assert prediction["processed_frame_count"] == 3
    assert sum(prediction["plane_counts"].values()) == 3
    assert prediction["abstained"] is True
    assert prediction["prediction_mode"] == "full_pilot_reference_fit"
    result = grade_prediction(
        prediction,
        thresholds=manifest.thresholds,
        decision_margin=manifest.decision_margin,
    ).to_dict()
    assert "classification" in result
    assert len(result["domain_status"]) == 7


def test_known_pilot_hash_excludes_matching_infant():
    manifest, _ = discover_model(ROOT / "models")
    assert manifest is not None and manifest.prototype_file
    model = load_model(ROOT / "models", manifest)
    payload = np.load(ROOT / "models" / manifest.prototype_file, allow_pickle=False)
    digest = str(payload["file_sha256"][0])
    expected_study = str(payload["study_codes"][0])
    prediction = prediction_to_json(model.predict(synthetic_frames(1), source_hashes=[digest]))
    assert prediction["prediction_mode"] == "patient_held_out_for_known_pilot_media"
    assert expected_study in prediction["matched_training_studies"]
