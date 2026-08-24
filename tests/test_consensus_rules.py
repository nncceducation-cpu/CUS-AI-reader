"""Rules taken directly from the 2021 Canadian consensus paper.

Each test names the rule it protects and where it comes from, so a future change
that breaks one is visibly a change to the clinical definition rather than a
refactor.
"""

from cus_ai.clinical import classify_study
from cus_ai.schemas import SideEvidence, StudyEvidence


def clean(side="right", **kwargs):
    defaults = dict(
        side=side,
        hemorrhage_present="no",
        adjacent_periventricular_echogenicity="no",
        clinician_verified=True,
    )
    defaults.update(kwargs)
    return SideEvidence(**defaults)


def bleeding(side="left", **kwargs):
    defaults = dict(
        side=side,
        hemorrhage_present="yes",
        confined_to_germinal_matrix="no",
        intraventricular_blood="yes",
        ventricular_distension="no",
        ahw_mm=4.0,
        adjacent_periventricular_echogenicity="no",
        clinician_verified=True,
    )
    defaults.update(kwargs)
    return SideEvidence(**defaults)


def test_echogenicity_not_brighter_than_choroid_is_physiologic():
    """Echodensity is defined against the choroid plexus.

    Peritrigonal blushes and symmetric frontal echodensities are normal in very
    preterm infants and must not be graded as PVHI.
    """
    evidence = StudyEvidence(
        study_code="A",
        left=bleeding(
            adjacent_periventricular_echogenicity="yes",
            echogenicity_brighter_than_choroid="no",
        ),
        right=clean(),
    )
    result = classify_study(evidence)
    assert result.left.pvhi == "Not present"
    assert any("physiologic" in item for item in result.left.warnings)


def test_focal_echogenicity_without_hemorrhage_is_significant_ischemic_injury():
    """Step 4, negative at Step 1: report as ischemic injury, not as PVHI."""
    evidence = StudyEvidence(
        study_code="B",
        left=clean(
            side="left",
            adjacent_periventricular_echogenicity="yes",
            echogenicity_brighter_than_choroid="yes",
        ),
        right=clean(),
    )
    result = classify_study(evidence)
    assert "significant ischemic injury" in result.left.pvhi
    assert result.left.gmh_ivh == "Negative for GMH-IVH"


def test_bilateral_echogenicity_with_only_small_hemorrhage_favours_ischemic_injury():
    """Bilateral white matter echogenicity coexisting with only a small GMH is
    more likely primary ischemic injury than bilateral PVHI."""
    side = dict(
        hemorrhage_present="yes",
        confined_to_germinal_matrix="yes",
        intraventricular_blood="no",
        adjacent_periventricular_echogenicity="yes",
        echogenicity_brighter_than_choroid="yes",
        clinician_verified=True,
    )
    evidence = StudyEvidence(
        study_code="C",
        left=SideEvidence(side="left", **side),
        right=SideEvidence(side="right", **side),
    )
    result = classify_study(evidence)
    assert result.left.gmh_ivh == "Grade I GMH-IVH"
    assert "ischemic" in result.left.pvhi
    assert "ischemic" in result.right.pvhi


def test_unilateral_echogenicity_with_grade_three_bleed_is_pvhi():
    evidence = StudyEvidence(
        study_code="D",
        left=bleeding(
            ventricular_distension="yes",
            ahw_mm=7.0,
            adjacent_periventricular_echogenicity="yes",
            echogenicity_brighter_than_choroid="yes",
        ),
        right=clean(),
    )
    result = classify_study(evidence)
    assert result.left.gmh_ivh == "Grade III GMH-IVH"
    assert result.left.pvhi == "Present"


def test_phvd_is_not_graded_inside_the_first_week():
    """Acute distension from a large bleed in the first days is grade III, not PHVD.

    The consensus separates the two deliberately, and PHVD is graded after the
    first week of age.
    """
    evidence = StudyEvidence(
        study_code="E",
        postnatal_age_days=3,
        left=bleeding(ventricular_distension="yes", ahw_mm=12.0),
        right=clean(),
        prior_gmh_ivh="yes",
        vi_above_97th="yes",
        vi_above_97th_plus_4mm="yes",
    )
    result = classify_study(evidence)
    assert "before day 7" in result.phvd
    assert result.left.gmh_ivh == "Grade III GMH-IVH"


def test_phvd_is_graded_after_the_first_week():
    evidence = StudyEvidence(
        study_code="F",
        postnatal_age_days=15,
        left=bleeding(ahw_mm=12.0),
        right=clean(),
        prior_gmh_ivh="yes",
        vi_above_97th="yes",
        vi_above_97th_plus_4mm="yes",
    )
    assert classify_study(evidence).phvd == "Severe PHVD"


def test_moderate_phvd_needs_both_vi_and_ahw_criteria():
    evidence = StudyEvidence(
        study_code="G",
        postnatal_age_days=20,
        left=bleeding(ahw_mm=8.0),
        right=clean(),
        prior_gmh_ivh="yes",
        vi_above_97th="yes",
        vi_above_97th_plus_4mm="no",
    )
    assert classify_study(evidence).phvd == "Moderate PHVD"


def test_moderate_cerebellar_hemorrhage_meets_the_severe_injury_definition():
    """Severe preterm brain injury includes moderate to large CBH."""
    evidence = StudyEvidence(
        study_code="H",
        left=clean(side="left"),
        right=clean(),
        cerebellar_hemorrhage="limited",
    )
    flag = classify_study(evidence).severe_preterm_brain_injury_flag
    assert "moderate cerebellar hemorrhage" in flag


def test_punctate_cerebellar_hemorrhage_does_not_meet_the_severe_definition():
    evidence = StudyEvidence(
        study_code="I",
        left=clean(side="left"),
        right=clean(),
        cerebellar_hemorrhage="punctate",
    )
    assert "No severe-injury criterion" in classify_study(evidence).severe_preterm_brain_injury_flag


def test_hemorrhage_off_the_gmh_ivh_axis_is_reported_not_silently_graded():
    evidence = StudyEvidence(
        study_code="J",
        left=bleeding(confined_to_germinal_matrix="no", intraventricular_blood="no"),
        right=clean(),
    )
    result = classify_study(evidence)
    assert result.left.gmh_ivh == "Hemorrhage outside the GMH-IVH axis"
    assert result.left.warnings


def test_missing_posterior_fossa_coverage_is_stated_as_a_limitation():
    evidence = StudyEvidence(
        study_code="K",
        left=clean(side="left"),
        right=clean(),
        coronal_views_complete=True,
        sagittal_views_complete=True,
        posterior_fossa_views_complete=False,
    )
    limitations = classify_study(evidence).limitations
    assert any("mastoid window" in item for item in limitations)
