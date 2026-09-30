from cus_ai.dataset_audit import audit_records


def record(infant="A", study="A-1", center="development", split="train", digest="a" * 64):
    return dict(infant_code=infant, study_code=study, center=center, split=split,
                source_sha256=[digest], adjudicated=True, labels={"left_gmh_ivh": "grade_1"})


def test_serial_exams_cannot_cross_partitions():
    result = audit_records([record(), record(study="A-2", split="internal_test", digest="b" * 64)])
    assert not result["valid"]
    assert any("Infant leakage" in item for item in result["errors"])


def test_duplicate_media_cannot_cross_partitions_under_different_infant_codes():
    result = audit_records([record(), record(infant="B", study="B-1", split="internal_test")])
    assert any("Duplicate-media leakage" in item for item in result["errors"])


def test_external_center_cannot_be_used_for_training():
    result = audit_records([record(), record(infant="B", study="B-1", split="external_test", digest="b" * 64)])
    assert any("External-center leakage" in item for item in result["errors"])


def test_independent_centers_and_infants_pass_structure_audit():
    result = audit_records([record(), record(infant="B", study="B-1", center="external", split="external_test", digest="b" * 64)])
    assert result["valid"]
    assert result["infants"] == 2
    assert result["warnings"]


def test_unadjudicated_labels_do_not_pass():
    item = record()
    item["adjudicated"] = False
    assert not audit_records([item])["valid"]
