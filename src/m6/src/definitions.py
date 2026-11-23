"""Frozen eligibility predicates (Protocol v3.0 §3). Pure functions, unit-tested in tests/test_definitions.py."""
import re
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "protocol.yaml").read_text())
ICD = CONFIG["icd"]
EICU = CONFIG["eicu"]


def _norm(code):
    return str(code).strip().upper().replace(".", "")


def icd_structural(code, version):
    """Structural intracranial traumatic injury (MIMIC format: no dots)."""
    c = _norm(code)
    if int(version) == 10:
        if not c.startswith(ICD["icd10_structural_prefix"]):
            return False
        is_extra = any(c.startswith(p) for p in ICD["icd10_extra_prefix"])
        if not is_extra and any(c.startswith(p) for p in ICD["icd10_structural_exclude_prefix"]):
            return False
        # 7th character encodes encounter; D/S are subsequent encounter / sequela.
        return not (len(c) >= 7 and c[6] in ICD["icd10_exclude_7th"])
    if int(version) == 9:
        if c[:3] in ICD["icd9_any"]:
            return True
        rule = ICD["icd9_4th_digit"]
        return c[:3] in rule["codes"] and len(c) >= 4 and c[3] in rule["digits"]
    return False


def icd_severity(code, version):
    """Severity-code route. S06/ICD-9 rules apply only to codes that are themselves structural."""
    c = _norm(code)
    sev = ICD["severity"]
    if int(version) == 10:
        if any(c.startswith(p) for p in sev["icd10_codes_prefix"]):
            return True
        if not icd_structural(c, 10):
            return False
        if c.startswith("S06A") and sev["icd10_s06a_all"]:
            return True
        return len(c) >= 6 and c[5] in sev["icd10_s06_6th"]
    if int(version) == 9:
        return icd_structural(c, 9) and len(c) >= 5 and c[4] in sev["icd9_5th"]
    return False


def eicu_icd9_tokens(field):
    """eICU icd9code mixes ICD-9 and truncated ICD-10, comma-separated. Return ICD-9 tokens only."""
    if field is None or field != field:
        return []
    toks = [t.strip() for t in str(field).split(",") if t.strip()]
    # ICD-9 diagnosis codes are numeric (V/E codes not relevant here); ICD-10 start with a letter.
    return [t for t in toks if re.fullmatch(r"\d{3}(\.\d{1,2})?", t)]


@lru_cache(maxsize=None)
def _rx(p):
    return re.compile(p, re.I)


def eicu_structural(path, icd9code):
    path = "" if path is None or path != path else str(path)
    if EICU["structural_path"].lower() in path.lower():
        return True
    if _rx(EICU["traumatic_hematoma_regex"]).search(path):
        return True
    return any(icd_structural(t, 9) for t in eicu_icd9_tokens(icd9code))


def eicu_severity(path, icd9code):
    path = "" if path is None or path != path else str(path)
    if _rx(EICU["severity_regex"]).search(path):
        return True
    return any(icd_severity(t, 9) for t in eicu_icd9_tokens(icd9code))


# ---------------------------------------------------------------- v3.1 additions
V31 = CONFIG["v31"]
_GCS_TXT = EICU["gcs_text_values"]
_GCS_RNG = EICU["gcs_ranges"]


def eicu_gcs_value(component, value):
    """Parse an eICU GCS component value (amendment C). Returns float or nan.

    Accepts a bare integer ('5'), an anchored value ('6-->(M6) obeys commands') or one of the
    fixed text labels; anything else, or a value outside the component's range, is missing.
    """
    if value is None or value != value:
        return float("nan")
    s = str(value).strip()
    m = re.match(r"^(\d+)(?:\s*-->|$)", s)
    v = float(m.group(1)) if m else float(_GCS_TXT.get(s, float("nan")))
    lo, hi = _GCS_RNG[component]
    return v if lo <= v <= hi else float("nan")


def death_side(t_h, L):
    """Timestamp indicator relative to landmark: 'le' (death <= L), 'gt', or None if unknown."""
    if t_h is None or t_h != t_h:
        return None
    return "le" if t_h <= L else "gt"


def date_side(day_start_h, L):
    """Date-only indicator (e.g. MIMIC dod): the death occurred somewhere in [start, start+24h)."""
    if day_start_h is None or day_start_h != day_start_h:
        return None
    if day_start_h + 24 <= L:
        return "le"
    if day_start_h > L:
        return "gt"
    return "ambiguous"


def classify_death(primary_h, primary_source, others, L):
    """Amendment F2. primary_h: recorded death time or flagged proxy (h since t0), or None when no
    death is recorded; others: sides ('le'/'gt'/'ambiguous'/None) from the remaining indicators.

    Returns (dead_before_L, unresolved).
    - Recorded/proxy death: dead before L only if it is <= L and no determinate indicator says > L;
      unresolved whenever determinate indicators disagree.
    - No recorded death: never dead before L, but unresolved if any indicator allows death <= L.
    """
    if primary_source is None:
        return False, any(s in ("le", "ambiguous") for s in others)
    det = {death_side(primary_h, L)} | {s for s in others if s in ("le", "gt")}
    det.discard(None)
    if len(det) > 1:
        return False, True
    return det == {"le"}, False


def sedation_marked(t, infusions, boluses, deep_rass):
    """Amendment E (MIMIC). infusions: list of (start_h, end_h); boluses / deep_rass: times (h).
    True if any marker is present at exam time t; False means only 'no identified marker'."""
    sd = V31["sedation"]
    if any(s <= t < e for s, e in infusions):
        return True
    if any(t - sd["bolus_lookback_h"] < b <= t for b in boluses):
        return True
    return any(t - sd["rass_lookback_h"] < r <= t for r in deep_rass)
