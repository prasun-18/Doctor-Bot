from safety.rules import evaluate_lab_risk


def assess_risk(report):

    risk_level, reasons = evaluate_lab_risk(report)

    return {
        "risk_level": risk_level,
        "reasons": reasons
    }