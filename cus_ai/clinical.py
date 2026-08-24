from __future__ import annotations

from .schemas import SideClassification, SideEvidence, StudyClassification, StudyEvidence


def _classify_side(e: SideEvidence) -> SideClassification:
    reasoning: list[str] = []
    warnings: list[str] = []

    if e.hemorrhage_present == "unknown":
        return SideClassification(
            side=e.side,
            gmh_ivh="Indeterminate",
            pvhi="Indeterminate",
            evidence_complete=False,
            reasoning=["Hemorrhage presence was not established."],
            warnings=["Review both coronal and parasagittal planes before classification."],
        )

    if e.hemorrhage_present == "no":
        reasoning.append("No germinal matrix or intraventricular hemorrhage was recorded.")
        if e.adjacent_periventricular_echogenicity == "yes":
            pvhi = "No PVHI: PVHI requires ipsilateral GMH-IVH"
            if e.echogenicity_brighter_than_choroid != "yes":
                warnings.append(
                    "Periventricular echogenicity requires comparison with the choroid plexus and serial evolution."
                )
        elif e.adjacent_periventricular_echogenicity == "no":
            pvhi = "Not present"
        else:
            pvhi = "Indeterminate"
        return SideClassification(
            side=e.side,
            gmh_ivh="Negative for GMH-IVH",
            pvhi=pvhi,
            evidence_complete=e.adjacent_periventricular_echogenicity != "unknown",
            reasoning=reasoning,
            warnings=warnings,
        )

    reasoning.append("Hemorrhage is present in or around the germinal matrix or ventricular system.")
    grade = "Indeterminate GMH-IVH grade"
    complete = False

    if e.confined_to_germinal_matrix == "yes":
        if e.intraventricular_blood == "yes":
            warnings.append("Confined germinal matrix hemorrhage conflicts with recorded intraventricular blood.")
        elif e.intraventricular_blood == "no":
            grade = "Grade I GMH-IVH"
            complete = True
            reasoning.append("Hemorrhage is confined to the germinal matrix with no intraventricular blood.")
        else:
            warnings.append("Intraventricular blood was not assessed.")
    elif e.confined_to_germinal_matrix == "no" or e.intraventricular_blood == "yes":
        if e.intraventricular_blood != "yes":
            warnings.append("Extension into the ventricular system was not explicitly confirmed.")
        elif e.ventricular_distension == "yes" and (
            (e.ahw_mm is not None and e.ahw_mm > 6) or e.ahw_above_6_mm == "yes"
        ):
            grade = "Grade III GMH-IVH"
            complete = True
            if e.ahw_mm is not None:
                reasoning.append(
                    f"Intraventricular blood acutely distends the ipsilateral ventricle and AHW is {e.ahw_mm:.1f} mm, above 6 mm."
                )
            else:
                reasoning.append(
                    "Intraventricular blood acutely distends the ipsilateral ventricle and the recorded AHW threshold is above 6 mm."
                )
        elif e.ventricular_distension == "unknown" or (
            e.ahw_mm is None and e.ahw_above_6_mm == "unknown"
        ):
            warnings.append("Grade II versus III requires acute ventricular distension and AHW measurement.")
        else:
            grade = "Grade II GMH-IVH"
            complete = True
            reasoning.append("Intraventricular blood is present without both grade III criteria.")
            if e.ventricular_distension == "yes" and e.ahw_mm is not None and e.ahw_mm <= 6:
                warnings.append("Distension is recorded, but AHW is not above 6 mm. The consensus step maps this to grade II.")
    else:
        warnings.append("Hemorrhage location was not established.")

    if e.adjacent_periventricular_echogenicity == "yes":
        pvhi = "Present"
        reasoning.append("Ipsilateral focal periventricular echogenicity accompanies GMH-IVH, meeting the PVHI rule.")
    elif e.adjacent_periventricular_echogenicity == "no":
        pvhi = "Not present"
    else:
        pvhi = "Indeterminate"
        complete = False
        warnings.append("PVHI cannot be excluded until adjacent periventricular white matter is assessed.")

    if not e.clinician_verified:
        warnings.append("Evidence has not been verified by a qualified clinician.")

    return SideClassification(
        side=e.side,
        gmh_ivh=grade,
        pvhi=pvhi,
        evidence_complete=complete,
        reasoning=reasoning,
        warnings=warnings,
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
    max_ahw = max([x for x in (e.left.ahw_mm, e.right.ahw_mm) if x is not None], default=None)
    ahw_above_6 = any(side.ahw_above_6_mm == "yes" for side in (e.left, e.right))
    ahw_above_10 = any(side.ahw_above_10_mm == "yes" for side in (e.left, e.right))
    if e.prior_gmh_ivh == "no":
        return "Not PHVD: no preceding GMH-IVH recorded"
    if e.prior_gmh_ivh == "unknown":
        return "Indeterminate: preceding GMH-IVH status is unknown"
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
    wmi = _classify_wmi(e)
    cbh = _classify_cbh(e.cerebellar_hemorrhage)
    phvd = _classify_phvd(e)

    severe_reasons: list[str] = []
    for side in (left, right):
        if side.gmh_ivh == "Grade III GMH-IVH" or side.pvhi == "Present":
            severe_reasons.append(f"{side.side}: {side.gmh_ivh}, PVHI {side.pvhi.lower()}")
    if e.wmi_pattern in {"grade_2", "grade_3", "grade_4"}:
        severe_reasons.append("cystic white matter injury")
    if e.cerebellar_hemorrhage == "large":
        severe_reasons.append("large cerebellar hemorrhage")
    if phvd == "Severe PHVD":
        severe_reasons.append("severe PHVD")

    limitations: list[str] = []
    if not e.all_frames_processed:
        limitations.append("Complete sequential processing of every reported source frame was not confirmed.")
    if not e.coronal_views_complete:
        limitations.append("Complete coronal sweep coverage was not confirmed.")
    if not e.sagittal_views_complete:
        limitations.append("Complete sagittal or parasagittal sweep coverage was not confirmed.")
    if not e.posterior_fossa_views_complete:
        limitations.append("Posterior fossa assessment was not confirmed.")
    if not e.serial_study_available:
        limitations.append("No serial study was confirmed. WMI evolution, PVHI cavitation, and PHVD trajectory may be missed.")
    if not (e.left.clinician_verified and e.right.clinician_verified):
        limitations.append("At least one hemisphere has not been clinician verified.")
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

