from pyhealth.data import Patient, Visit


def drug_recommendation_mimic4_fn(patient: Patient):
    """
    Generate drug recommendation samples for each patient in MIMIC-IV:
      - Extract diagnosis, procedure, prescription codes per visit
      - Map prescriptions to ATC level-3 by taking first 4 characters
      - Exclude visits without all three code types or patients with <2 valid visits
      - Accumulate historical visits as context
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        drugs = [d[:4] for d in drugs]
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
            "drugs_all": drugs,
        })

    if len(samples) < 2:
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs_all"] = samples[i-1]["drugs_all"] + [samples[i]["drugs_all"]]

    return samples


def mortality_prediction_mimic4_fn(patient: Patient):
    """
    Generate mortality prediction samples for MIMIC-IV:
      - Use visits 0..n-2 to predict next visit's discharge_status (0 or 1)
      - Exclude visits missing codes
      - Accumulate history
    """
    samples = []
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i+1]
        label = int(next_visit.discharge_status) if next_visit.discharge_status in (0,1) else 0
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
            "label": label,
        })

    if not samples:
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i-1]["drugs"] + [samples[i]["drugs"]]

    return samples


def readmission_prediction_mimic4_fn(patient: Patient, time_window: int = 15):
    """
    Generate readmission prediction samples for MIMIC-IV:
      - Predict if next visit occurs within `time_window` days
      - Exclude visits missing codes
      - Accumulate history
    """
    samples = []
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i+1]
        td = (next_visit.encounter_time - visit.encounter_time).days
        label = 1 if td < time_window else 0
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
            "label": label,
        })

    if not samples:
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i-1]["drugs"] + [samples[i]["drugs"]]

    return samples


def categorize_los(days: int) -> int:
    """
    Map length-of-stay days into 10 categories: <1 day=0, 1–7 days=1–7, 8–14 days=8, >14=9
    """
    if days < 1:
        return 0
    elif days <= 7:
        return days
    elif days <= 14:
        return 8
    else:
        return 9


def length_of_stay_prediction_mimic4_fn(patient: Patient):
    """
    Generate length-of-stay prediction samples for MIMIC-IV:
      - Compute LOS days for each visit, map to category
      - Exclude visits missing codes
      - Accumulate history
    """
    samples = []
    for visit in patient:
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        days = (visit.discharge_time - visit.encounter_time).days
        label = categorize_los(days)
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
            "label": label,
        })

    if not samples:
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i-1]["drugs"] + [samples[i]["drugs"]]

    return samples
