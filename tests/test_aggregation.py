from cus_ai.aggregation import AggregationConfig, aggregate_labels, frame_weight


def sweep(values, plane="coronal", source="clip.avi"):
    return [
        {
            "plane": plane,
            "source_name": source,
            "frame_index": index,
            "weight": 1.0,
            "probabilities": {"finding": value},
        }
        for index, value in enumerate(values)
    ]


def score(rows, **kwargs):
    config = AggregationConfig(require_two_planes=False, **kwargs)
    return aggregate_labels(rows, ["finding"], config)["finding"]


def test_isolated_high_frame_does_not_carry_a_finding():
    """One bright frame in a long sweep is speckle, not a clot.

    Under the previous top-k mean this study scored 0.36 from a single frame and
    a run of three such frames scored 0.99.
    """
    values = [0.05] * 200
    values[50] = 0.99
    assert score(sweep(values)).probability < 0.10


def test_scattered_high_frames_do_not_carry_a_finding():
    values = [0.05] * 200
    for index in (20, 90, 150):
        values[index] = 0.98
    result = score(sweep(values))
    assert result.probability < 0.10
    assert result.longest_run < 3


def test_sustained_run_carries_a_finding():
    values = [0.05] * 200
    values[40:58] = [0.90] * 18
    result = score(sweep(values))
    assert result.probability > 0.85
    assert result.longest_run >= 18


def test_still_image_study_is_not_zeroed_by_the_run_requirement():
    """Centres that export single stills must still be gradeable."""
    result = score(sweep([0.92]), min_run=3)
    assert result.probability > 0.85


def test_second_plane_confirmation_moderates_a_single_plane_finding():
    coronal = sweep([0.05] * 40, plane="coronal")
    coronal[10:20] = [
        {**row, "probabilities": {"finding": 0.95}} for row in coronal[10:20]
    ]
    sagittal = sweep([0.05] * 40, plane="sagittal", source="sag.avi")
    config = AggregationConfig(require_two_planes=True)
    single_plane = aggregate_labels(coronal, ["finding"], config)["finding"]
    both_planes = aggregate_labels(coronal + sagittal, ["finding"], config)["finding"]
    assert both_planes.probability < single_plane.probability
    assert "coronal" in both_planes.planes_used and "sagittal" in both_planes.planes_used


def test_labels_are_restricted_to_planes_that_show_the_anatomy():
    """A cerebellar label scored off a coronal frontal-horn frame is not evidence."""
    rows = sweep([0.95] * 30, plane="coronal")
    rows = [{**row, "probabilities": {"cbh_large": 0.95}} for row in rows]
    result = aggregate_labels(rows, ["cbh_large"], AggregationConfig())["cbh_large"]
    assert result.in_scope is False
    assert result.probability == 0.0


def test_poor_quality_frames_are_down_weighted():
    good = frame_weight(0.95, {"quality_flag": "reviewable", "clipped_fraction": 0.0})
    blurred = frame_weight(0.95, {"quality_flag": "possible blur", "clipped_fraction": 0.0})
    clipped = frame_weight(0.95, {"quality_flag": "reviewable", "clipped_fraction": 0.5})
    assert blurred < good and clipped < good
