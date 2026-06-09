"""
src/report.py
=============
Structured clinical text report from ILO + PMF model outputs.
Matches your notebook Cell 18 report format.
"""

import yaml

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_ILO_DESC = {
    "0/-": "Sub-threshold — no pneumoconiotic opacities detectable.",
    "0/0": "No opacities present. Normal reference category.",
    "0/1": "Borderline — between categories 0 and 1.",
    "1/0": "Borderline — between categories 1 and 0.",
    "1/1": "Category 1 — small opacities present. Mild silicosis pattern.",
    "1/2": "Between categories 1 and 2. Mild–moderate profusion.",
    "2/1": "Between categories 2 and 1. Moderate silicosis.",
    "2/2": "Category 2 — moderate profusion of small opacities.",
    "2/3": "Between categories 2 and 3. Dense opacities with coalescence.",
    "3/2": "Between categories 3 and 2. Advanced silicosis.",
    "3/3": "Category 3 — very high profusion. Severe silicosis.",
    "3/+": "Beyond category 3. End-stage / very severe silicosis.",
}

_PMF_DESC = {
    "None": "No large opacities. PMF absent.",
    "A":    "PMF Grade A: single opacity 1–5 cm.",
    "B":    "PMF Grade B: opacities ≤ area of right upper zone.",
    "C":    "PMF Grade C: opacities > area of right upper zone.",
}

_URGENCY = {
    "0/-": "Routine", "0/0": "Routine", "0/1": "Routine",
    "1/0": "Monitor", "1/1": "Monitor", "1/2": "Monitor",
    "2/1": "Refer",   "2/2": "Refer",   "2/3": "Refer",
    "3/2": "Urgent",  "3/3": "Urgent",  "3/+": "Urgent",
}


def generate_report(ilo_class: str, pmf_class: str,
                    confidence: float,
                    ilo_probs: list = None,
                    pmf_probs: list = None) -> str:
    ilo_desc  = _ILO_DESC.get(ilo_class, f"ILO {ilo_class}")
    pmf_desc  = _PMF_DESC.get(pmf_class,  f"PMF {pmf_class}")
    urgency   = _URGENCY.get(ilo_class,   "Unknown")
    conf_pct  = round(confidence * 100, 1)

    # Top-2 ILO differential
    diff = ""
    if ilo_probs:
        labels = _cfg["ilo_classes"]
        ranked = sorted(enumerate(ilo_probs), key=lambda x: x[1], reverse=True)
        others = [(labels[i], round(p*100,1)) for i,p in ranked if labels[i] != ilo_class][:2]
        if others:
            diff = "\nDIFFERENTIAL:\n" + "".join(f"  {l}: {p}%\n" for l,p in others)

    return (
        "=" * 55 + "\n"
        "rises.AI — SILICOSIS DIAGNOSTIC REPORT\n"
        "=" * 55 + "\n\n"
        f"ILO PROFUSION SCORE : {ilo_class}\n"
        f"{ilo_desc}\n\n"
        f"PMF GRADE           : {pmf_class}\n"
        f"{pmf_desc}\n\n"
        f"CONFIDENCE          : {conf_pct}%\n"
        f"RECOMMENDED ACTION  : {urgency}\n"
        f"{diff}\n"
        "=" * 55 + "\n"
        "DISCLAIMER: AI output for RESEARCH PURPOSES ONLY.\n"
        "Must be reviewed by a qualified radiologist.\n"
        "=" * 55
    )
