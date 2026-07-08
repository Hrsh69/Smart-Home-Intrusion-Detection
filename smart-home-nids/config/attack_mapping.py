"""
Attack label mapping for CIC IoT-2023.

The raw CIC IoT-2023 labels are fine-grained (e.g., DDoS-UDP_Flood, Recon-PortScan).
For a practical smart-home NIDS (and for stable deployment/inference), we map them to
broader categories. This file is intentionally isolated so future labels can be added
without touching pipeline code.
"""

from __future__ import annotations

import re
from typing import Dict


# Broad categories expected by the project.
KNOWN_CATEGORIES = [
    "BENIGN",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "BruteForce",
    "WebAttack",
    "Malware",
    "Unknown",
]


def _compile_rules() -> Dict[str, re.Pattern]:
    # Regex rules (case-insensitive) to map fine-grained labels to categories.
    # Ordered by specificity.
    rules: Dict[str, str] = {
        "BENIGN": r"^(benign|benigntraffic)$",
        "DDoS": r"^ddos[\-_].*",
        "DoS": r"^dos[\-_].*",
        "Mirai": r"^mirai[\-_].*",
        "Recon": r"^recon[\-_].*|^vulnerabilityscan$",
        "Spoofing": r"spoof|^mitm[\-_].*|arp",
        "BruteForce": r"bruteforce|dictionary",
        "WebAttack": r"(sqlinjection|xss|commandinjection|uploading[_-]attack|browserhijacking)",
        "Malware": r"(backdoor[_-]malware|malware)",
    }
    return {k: re.compile(v, flags=re.IGNORECASE) for k, v in rules.items()}


_RULES = _compile_rules()


def map_attack_label(raw_label: str) -> str:
    """
    Map a CIC IoT-2023 raw label to a broader category.

    Args:
        raw_label: The raw label string from the dataset (e.g., "DDoS-UDP_Flood").

    Returns:
        Broad attack category (one of KNOWN_CATEGORIES).
    """
    if raw_label is None:
        return "Unknown"

    s = str(raw_label).strip()
    if not s:
        return "Unknown"

    for category, pattern in _RULES.items():
        if pattern.search(s):
            return category

    return "Unknown"

