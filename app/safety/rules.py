def evaluate_lab_risk(report):

    risk_level = "LOW"
    reasons = []

    for lab in report.lab_values:

        if lab.test_name.lower() in ["hemoglobin", "hb"]:
            if lab.value < 7:
                risk_level = "HIGH"
                reasons.append("Critically low hemoglobin")
            elif lab.value < 10 and risk_level != "HIGH":
                risk_level = "MODERATE"
                reasons.append("Low hemoglobin")

        if lab.is_abnormal:
            if risk_level == "LOW":
                risk_level = "MODERATE"

    return risk_level, reasons