"""Deterministic consensus rule engine.

Implements the four-step GMH-IVH algorithm, PVHI, ischemic white matter injury,
cerebellar hemorrhage, PHVD, and the severe-injury definition from:

    Mohammad K, Scott JN, Leijser LM, et al. Consensus Approach for
    Standardizing the Screening and Classification of Preterm Brain Injury
    Diagnosed With Cranial Ultrasound: A Canadian Perspective.
    Front Pediatr. 2021;9:618236. doi:10.3389/fped.2021.618236

Rules that this module now enforces and previously did not:

* Echodensity is defined against the choroid plexus. White matter echogenicity
  that does not exceed choroid brightness is physiologic and is not scored as
  PVHI or as PVE.
* Step 4 asks about echogenicity ipsilateral to the GMH-IVH, or in bilateral
  disease ipsilateral to the *largest* GMH-IVH. Bilateral ill-defined
  echogenicity coexisting with only small germinal matrix hemorrhage is
  ischemic white matter injury, not PVHI.
* Focal echogenicity with no GMH-IVH is reported as significant ischemic injury.
* Blood in the third or fourth ventricle satisfies Step 1.
* PHVD is graded on the largest lateral ventricle after the first week of age.
  Acute distension from a grade III bleed inside the first week is not PHVD.
* Severe preterm brain injury includes moderate as well as large cerebellar
  hemorrhage.
"""

from __future__ import annotations

from .schemas import SideClassification, SideEvidence, StudyClassification, StudyEvidence

GRADE_SEVERITY = {
    "Grade I GMH-IVH": 1,
    "Grade II GMH-IVH": 2,
    "Grade III GMH-IVH": 3,
}

PHVD_EARLIEST_DAY = 7


def _classify_cystic_sequela(value: str) -> str:
    labels = {
        "none": "No porencephalic cyst recorded",
        "porencephalic": "Porencephalic cyst, consistent with evolved PVHI",
        "multiple_evolved_pvhi": "Multiple unilateral cysts, consistent with evolved PVHI",
        "not_assessed": "Not assessed",
    }
    return labels.get(value, "Indeterminate cystic sequela")


def _abnormal_echogenicity(e: SideEvidence) -> str:
    """Is periventricular echogenicity present and abnormal by the choroid rule?

    The consensus defines echodensity as brightness exceeding the choroid
    plexus. Echogenicity that is explicitly not brighter than choroid is
    physiologic, most often a peritrigonal blush or a symmetric frontal
    echodensity, and must not be graded as injury.
    """
    if e.adjacent_periventricular_echogenicity == "no":
        return "no"
    if e.adjacent_periventricular_echogenicity == "unknown":
        return "unknown"
    if e.echogenicity_brighter_than_choroid == "no":
        return "physiologic"
    return "yes"


def _classify_side(e: SideEvidence) -> SideClassification:
    reasoning: list[str] = []
    warnings: list[str] = []
    cystic_sequela = _classify_cystic_sequela(e.cystic_change)
    evolved_pvhi = e.cystic_change in {"porencephalic", "multiple_evolved_pvhi"}
    echogenicity = _abnormal_echogenicity(e)

    if echogenicity == "physiologic":
        warnings.append(
            "Periventricular echogenicity does not exceed choroid plexus brightness and is "
            "recorded as physiologic."
        )

    if e.hemorrhage_present == "unknown":
        unknown_reasoning = ["Hemorrhage presence was not established."]
        if evolved_pvhi:
            unknown_reasoning.append(
                "A cystic lesion was recorded as the ipsilateral evolution of previous PVHI."
            )
        return SideClassification(
            side=e.side,
            gmh_ivh="Indeterminate",
            pvhi="Evolved PVHI with porencephalic cyst" if evolved_pvhi else "Indeterminate",
            cystic_sequela=cystic_sequela,
            evidence_complete=False,
            reasoning=unknown_reasoning,
            warnings=["Review both coronal and parasagittal planes before classification."],
        )

    # Step 1 answered NO: negative for GMH-IVH, straight to Step 4.
    if e.hemorrhage_present == "no":
        reasoning.append(
            "No hemorrhage was recorded in or around the germinal matrix or within the "
            "lateral, third, or fourth ventricle."
        )
        if evolved_pvhi:
            pvhi = "Evolved PVHI with porencephalic cyst"
            reasoning.append(
                "The porencephalic cyst records the serial evolution of previous ipsilateral PVHI."
            )
        elif echogenicity == "yes":
            pvhi = "No PVHI: significant ischemic injury without GMH-IVH"
            reasoning.append(
                "Focal periventricular echogenicity without ipsilateral GMH-IVH is reported as "
                "significant ischemic injury at Step 4."
            )
            if e.echogenicity_brighter_than_choroid != "yes":
                warnings.append(
                    "Confirm that the echogenicity exceeds choroid plexus brightness and follow "
                    "its serial evolution."
                )
        elif echogenicity in {"no", "physiologic"}:
            pvhi = "Not present"
        else:
            pvhi = "Indeterminate"
        return SideClassification(
            side=e.side,
            gmh_ivh="Negative for GMH-IVH",
            pvhi=pvhi,
            cystic_sequela=cystic_sequela,
            evidence_complete=evolved_pvhi or echogenicity != "unknown",
            reasoning=reasoning,
            warnings=warnings,
        )

    # Step 1 answered YES.
    reasoning.append(
        "Hemorrhage is present in or around the germinal matrix or within the ventricular system."
    )
    grade = "Indeterminate GMH-IVH grade"
    complete = False

    if e.confined_to_germinal_matrix == "yes":
        if e.intraventricular_blood == "yes":
            warnings.append(
                "Confined germinal matrix hemorrhage conflicts with recorded intraventricular blood."
            )
        elif e.intraventricular_blood == "no":
            grade = "Grade I GMH-IVH"
            complete = True
            reasoning.append(
                "Hemorrhage is confined to the germinal matrix with no blood in the lateral "
                "ventricle or on the choroid plexus."
            )
        else:
            warnings.append("Intraventricular blood was not assessed.")
    elif e.confined_to_germinal_matrix == "no" or e.intraventricular_blood == "yes":
        if e.intraventricular_blood == "no":
            # Hemorrhage that is neither confined to the germinal matrix nor
            # inside the ventricle is not on the Papile-Volpe axis at all. It is
            # reported rather than silently graded.
            grade = "Hemorrhage outside the GMH-IVH axis"
            complete = True
            reasoning.append(
                "Hemorrhage was recorded as neither confined to the germinal matrix nor within "
                "the ventricular system, so the four-step GMH-IVH grades do not apply."
            )
            warnings.append(
                "Confirm the location of the hemorrhage. Consider cerebellar, subdural, or "
                "parenchymal sources."
            )
        elif e.intraventricular_blood != "yes":
            warnings.append("Extension into the ventricular system was not explicitly confirmed.")
        elif e.ventricular_distension == "yes" and (
            (e.ahw_mm is not None and e.ahw_mm > 6) or (e.ahw_mm is None and e.ahw_above_6_mm == "yes")
        ):
            grade = "Grade III GMH-IVH"
            complete = True
            if e.ahw_mm is not None:
                reasoning.append(
                    f"Intraventricular blood acutely distends the ipsilateral ventricle and AHW is "
                    f"{e.ahw_mm:.1f} mm, above 6 mm."
                )
            else:
                reasoning.append(
                    "Intraventricular blood acutely distends the ipsilateral ventricle and the "
                    "recorded AHW threshold is above 6 mm."
                )
        elif e.ventricular_distension == "unknown" or (
            e.ahw_mm is None and e.ahw_above_6_mm == "unknown"
        ):
            warnings.append(
                "Grade II versus III requires acute ventricular distension and an AHW measurement "
                "in the coronal plane at the foramen of Monro."
            )
        else:
            grade = "Grade II GMH-IVH"
            complete = True
            reasoning.append(
                "Intraventricular blood is present without both grade III criteria. Clot "
                "characteristically fills less than half the lateral ventricle."
            )
            if e.ventricular_distension == "yes" and e.ahw_mm is not None and e.ahw_mm <= 6:
                warnings.append(
                    "Distension is recorded, but AHW is not above 6 mm. Step 3 maps this to grade II."
                )
    else:
        warnings.append("Hemorrhage location was not established.")

    # Step 4 for a side that is positive at Step 1.
    if echogenicity == "yes":
        pvhi = "PVHI with porencephalic cyst" if evolved_pvhi else "Present"
        reasoning.append(
            "Ipsilateral focal periventricular echogenicity accompanies GMH-IVH, meeting the "
            "Step 4 rule for PVHI."
        )
        if e.echogenicity_brighter_than_choroid != "yes":
            warnings.append(
                "Confirm that the echogenicity exceeds choroid plexus brightness before "
                "recording PVHI."
            )
    elif evolved_pvhi:
        pvhi = "Evolved PVHI with porencephalic cyst"
        reasoning.append("The cystic lesion records the serial evolution of previous ipsilateral PVHI.")
    elif echogenicity in {"no", "physiologic"}:
        pvhi = "Not present"
    else:
        pvhi = "Indeterminate"
        complete = False
        warnings.append(
            "PVHI cannot be excluded until adjacent periventricular white matter is assessed."
        )

    if not e.clinician_verified:
        warnings.append("Evidence has not been verified by a qualified clinician.")

    return SideClassification(
        side=e.side,
        gmh_ivh=grade,
        pvhi=pvhi,
        cystic_sequela=cystic_sequela,
        evidence_complete=complete,
        reasoning=reasoning,
        warnings=warnings,
    )


def _apply_bilateral_step_four(
    e: StudyEvidence, left: SideClassification, right: SideClassification
) -> None:
    """Resolve Step 4 when both hemispheres carry findings.

    Step 4 is asked of the side of the GMH-IVH, or in bilateral disease of the
    side with the largest GMH-IVH. The consensus also warns that bilateral areas
    of white matter echogenicity brighter than the choroid, present with no blood
    or with only a small germinal matrix hemorrhage, are more likely primary
    ischemic injury than PVHI. PVHI is characteristically fan-shaped, unilateral,
    and paired with a substantial ipsilateral bleed.
    """
    left_abnormal = _abnormal_echogenicity(e.left) == "yes"
    right_abnormal = _abnormal_echogenicity(e.right) == "yes"
    if not (left_abnormal and right_abnormal):
        return

    left_grade = GRADE_SEVERITY.get(left.gmh_ivh, 0)
    right_grade = GRADE_SEVERITY.get(right.gmh_ivh, 0)
    if max(left_grade, right_grade) == 0:
        return

    # Bilateral echogenicity with only small (grade I) hemorrhage favours
    # ischemic white matter injury over bilateral PVHI.
    if max(left_grade, right_grade) <= 1:
        note = (
            "Bilateral periventricular echogenicity coexisting with only small germinal matrix "
            "hemorrhage favours primary ischemic white matter injury over PVHI. Serial imaging is "
            "required to separate the two."
        )
        for side in (left, right):
            if side.pvhi == "Present":
                side.pvhi = "Indeterminate: bilateral pattern favours ischemic white matter injury"
                side.warnings.append(note)
                side.evidence_complete = False
        return

    # Asymmetric disease: Step 4 is anchored to the larger hemorrhage. The
    # smaller side keeps its finding but is flagged for the ischemic differential.
    if left_grade != right_grade:
        smaller = left if left_grade < right_grade else right
        larger = right if left_grade < right_grade else left
        if smaller.pvhi == "Present":
            smaller.warnings.append(
                f"Step 4 anchors to the larger {larger.side} GMH-IVH. Confirm that the "
                f"{smaller.side} echogenicity is fan-shaped and asymmetric before recording a "
                "second PVHI rather than coexisting ischemic injury."
            )


def _classify_wmi(e: StudyEvidence) -> str:
    labels = {
        "none": "No ischemic WMI pattern recorded",
        "pve_under_7_days": "PVE present, duration below 7 days or not yet established",
        "grade_1": "Grade 1 WMI: PVE persisting for 7 days or longer without cystic evolution",
        "grade_2": "Grade 2 WMI: PVE evolving into small localized frontoparietal cysts",
        "grade_3": "Grade 3 WMI: PVE evolving into extensive fronto-parieto-occipital periventricular cysts",
        "grade_4": "Grade 4 WMI: PVE evolving into extensive deep or subcortical white matter cysts",
        "not_assessed": "Not assessed",
    }
    return labels.get(e.wmi_pattern, "Indeterminate WMI pattern")


def _classify_cbh(value: str) -> str:
    labels = {
        "none": "No cerebellar hemorrhage recorded",
        "punctate": "Punctate CBH, 4 mm or smaller, usually MRI detected",
        "limited": "Limited CBH, above 4 mm and below one third of a cerebellar hemisphere",
        "large": "Large CBH, one third or more of a cerebellar hemisphere",
        "not_assessed": "Not assessed",
    }
    return labels.get(value, "Indeterminate cerebellar hemorrhage category")


def _classify_phvd(e: StudyEvidence) -> str:
    """Grade PHVD on the largest lateral ventricle, after the first week of age.

    Moderate is VI above the 97th centile and AHW above 6 mm. Severe is VI above
    the 97th centile plus 4 mm, or AHW above 10 mm. Acute distension from a large
    hemorrhage in the first days after birth is grade III GMH-IVH, not PHVD, and
    the consensus separates the two deliberately.
    """
    max_ahw = max([x for x in (e.left.ahw_mm, e.right.ahw_mm) if x is not None], default=None)
    ahw_above_6 = any(side.ahw_above_6_mm == "yes" for side in (e.left, e.right))
    ahw_above_10 = any(side.ahw_above_10_mm == "yes" for side in (e.left, e.right))

    if e.prior_gmh_ivh == "no":
        if e.vi_above_97th == "no" and not ahw_above_6 and (max_ahw is None or max_ahw <= 6):
            return "No moderate or severe PHVD by recorded thresholds"
        return "Not PHVD: no preceding GMH-IVH recorded"
    if e.prior_gmh_ivh == "unknown":
        return "Indeterminate: preceding GMH-IVH status is unknown"

    age = e.postnatal_age_days
    if age is not None and age < PHVD_EARLIEST_DAY:
        return (
            "Not gradeable as PHVD before day 7: acute ventricular distension in the first week "
            "is graded as GMH-IVH"
        )

    if e.vi_above_97th_plus_4mm == "yes" or (max_ahw is not None and max_ahw > 10) or ahw_above_10:
        return "Severe PHVD"
    if e.vi_above_97th == "yes" and ((max_ahw is not None and max_ahw > 6) or ahw_above_6):
        return "Moderate PHVD"
    if e.vi_above_97th == "no" and (max_ahw is None or max_ahw <= 6):
        return "No moderate or severe PHVD by recorded thresholds"
    return "Indeterminate PHVD: complete serial VI and AHW measurements are required"


def classify_study(e: StudyEvidence) -> StudyClassification:
    left = _classify_side(e.left)
    right = _classify_side(e.right)
    _apply_bilateral_step_four(e, left, right)
    wmi = _classify_wmi(e)
    cbh = _classify_cbh(e.cerebellar_hemorrhage)
    phvd = _classify_phvd(e)

    severe_reasons: list[str] = []
    for side in (left, right):
        if side.gmh_ivh == "Grade III GMH-IVH" or side.pvhi in {
            "Present",
            "PVHI with porencephalic cyst",
            "Evolved PVHI with porencephalic cyst",
        }:
            severe_reasons.append(f"{side.side}: {side.gmh_ivh}, PVHI {side.pvhi.lower()}")
    if e.wmi_pattern in {"grade_2", "grade_3", "grade_4"}:
        severe_reasons.append("cystic white matter injury")
    # The consensus defines severe injury as including moderate to large CBH.
    if e.cerebellar_hemorrhage == "limited":
        severe_reasons.append("moderate cerebellar hemorrhage")
    if e.cerebellar_hemorrhage == "large":
        severe_reasons.append("large cerebellar hemorrhage")
    if phvd == "Severe PHVD":
        severe_reasons.append("severe ventricular dilatation")

    limitations: list[str] = []
    if not e.all_frames_processed:
        limitations.append("Complete sequential processing of every reported source frame was not confirmed.")
    if not e.coronal_views_complete:
        limitations.append("Complete coronal sweep coverage was not confirmed.")
    if not e.sagittal_views_complete:
        limitations.append("Complete sagittal or parasagittal sweep coverage was not confirmed.")
    if not e.posterior_fossa_views_complete:
        limitations.append(
            "Posterior fossa assessment through the mastoid window was not confirmed, so "
            "cerebellar hemorrhage and fourth ventricular blood may be missed."
        )
    if not e.serial_study_available:
        limitations.append(
            "No serial study was confirmed. WMI evolution, PVHI cavitation, and PHVD trajectory may be missed."
        )
    if not (e.left.clinician_verified and e.right.clinician_verified):
        limitations.append("At least one hemisphere has not been clinician verified.")
    if e.postnatal_age_days is None:
        limitations.append(
            "Postnatal age was not supplied, so the first-week grading window and the day 7 PHVD "
            "boundary could not be applied."
        )
    if e.model_plane_counts:
        if e.model_plane_counts.get("coronal", 0) == 0:
            limitations.append("The installed model accepted no coronal frame.")
        if e.model_plane_counts.get("sagittal", 0) == 0:
            limitations.append("The installed model accepted no sagittal frame.")

    core_complete = (
        e.all_frames_processed
        and e.complete_required_views
        and e.left.clinician_verified
        and e.right.clinician_verified
        and left.evidence_complete
        and right.evidence_complete
    )
    serial_domains_complete = (
        e.serial_study_available
        and e.wmi_pattern != "not_assessed"
        and e.cerebellar_hemorrhage != "not_assessed"
        and e.prior_gmh_ivh != "unknown"
        and e.vi_above_97th != "unknown"
        and e.vi_above_97th_plus_4mm != "unknown"
    )
    classification_status = (
        "Final consensus classification from complete verified study and serial evidence"
        if core_complete and serial_domains_complete
        else "Provisional consensus classification"
    )

    return StudyClassification(
        left=left,
        right=right,
        wmi=wmi,
        cerebellar_hemorrhage=cbh,
        phvd=phvd,
        classification_status=classification_status,
        view_coverage={
            "coronal": e.coronal_views_complete,
            "sagittal_or_parasagittal": e.sagittal_views_complete,
            "posterior_fossa": e.posterior_fossa_views_complete,
        },
        severe_preterm_brain_injury_flag=(
            "Criteria recorded: " + "; ".join(severe_reasons)
            if severe_reasons
            else "No severe-injury criterion established from the recorded evidence"
        ),
        limitations=limitations,
    )
