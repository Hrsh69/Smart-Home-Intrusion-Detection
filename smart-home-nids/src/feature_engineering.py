from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import safe_div


def engineer_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create meaningful cybersecurity features for flow-based NIDS.

    The CIC IoT-2023 CSV (as provided in this workspace) includes flow stats like:
    - flow_duration, duration
    - tot_size, number (packet count proxy), header_length
    - rate/srate/drate
    - per-flag counts and flag-number indicators
    These engineered features aim to capture burstiness, intensity and control-plane behavior.
    """
    out = df.copy()

    # Normalize column spellings seen in CIC IoT-2023 exports.
    cols = set(out.columns)
    flow_dur = "flow_duration" if "flow_duration" in cols else None
    duration = "duration" if "duration" in cols else None
    tot_size = "tot_size" if "tot_size" in cols else ("tot_sum" if "tot_sum" in cols else None)
    pkt_count = "number" if "number" in cols else None

    # Prefer flow_duration if present; fallback to duration.
    time_col = flow_dur or duration

    if time_col and pkt_count:
        out["packets_per_second"] = safe_div(out[pkt_count], out[time_col])

    if tot_size and pkt_count:
        out["bytes_per_packet"] = safe_div(out[tot_size], out[pkt_count])
        out["avg_packet_size"] = out["bytes_per_packet"]

    if time_col and tot_size:
        out["bytes_per_second"] = safe_div(out[tot_size], out[time_col])
        out["flow_rate_bps"] = out["bytes_per_second"]

    # Flag ratios: these can be very indicative of scans, SYN floods, reset storms, etc.
    flag_counts = [
        c
        for c in out.columns
        if c.endswith("_count") and any(k in c for k in ("syn", "ack", "fin", "rst", "urg"))
    ]
    total_flags = None
    if flag_counts:
        out["total_flag_count"] = out[flag_counts].sum(axis=1, skipna=True)
        total_flags = "total_flag_count"

    for flag in ("syn", "ack", "fin", "rst", "urg"):
        c = f"{flag}_count"
        if c in out.columns and total_flags:
            out[f"{flag}_ratio"] = safe_div(out[c], out[total_flags])

    # Directionality ratio when Srate/Drate exist (source vs destination rates).
    if "srate" in cols and "drate" in cols:
        out["traffic_direction_ratio"] = safe_div(out["srate"], out["drate"])

    # Burstiness proxy: std / (avg + eps)
    if "std" in cols and "avg" in cols:
        out["burstiness"] = safe_div(out["std"], out["avg"])

    # Header intensity: header bytes per packet (if header length exists)
    if "header_length" in cols and pkt_count:
        out["header_bytes_per_packet"] = safe_div(out["header_length"], out[pkt_count])

    # Sanitize any infs produced by divisions
    out = out.replace([np.inf, -np.inf], np.nan)

    logger.info("Feature engineering added %d columns.", len(set(out.columns) - set(df.columns)))
    return out

