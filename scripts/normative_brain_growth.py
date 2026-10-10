"""Courbes de croissance cérébrale normatives (volume ↔ âge).

Ce module fournit une référence normative *unifiée* couvrant le fœtal (par âge
gestationnel, GA) **et** le postnatal (par mois), et — crucialement — l'inverse
``volume → âge`` qui permet une estimation d'âge robuste **décorrélée de
l'alignement atlas** (cf. RESEARCH_SOTA_STRATEGY.md §3.4 ; brise la circularité du
PATCH P19).

Axe d'âge unifié : on exprime tout en « mois postnatals », le terme (40 SG)
valant le mois 0. Un fœtus de 28 SG vaut donc ``(28 - 40) / 4.345 ≈ -2.76`` mois.
Cela permet de réutiliser exactement la machinerie d'interpolation mensuelle
existante (``_apply_normative_volume_bounds``, PATCH P12) en l'étendant vers les
âges négatifs (fœtal).

Conçu **sans dépendance lourde** (Python pur) pour être trivialement testable.

Sources des bandes ±2σ :
- Postnatal : Knickmeyer 2008, Gilmore 2012, Peterson 2003, dHCP 2018 — la table
  postnatale reflète ``adaptive_mask_fusion._NORMATIVE_BRAIN_VOLUME_CM3`` (un test
  vérifie l'égalité pour éviter toute dérive).
- Fœtal : volumétrie IRM fœtale (Andescavage 2017, Jarvis 2016,
  Kyriakopoulou 2017 ; AJNR 44(1):82). Calibré sur la plage de masques observée
  pour FetalFMRI (~40-570 cm³).
"""

from __future__ import annotations

from typing import Mapping

# Conversion semaines de gestation ↔ mois postnatals (terme = 40 SG = mois 0).
_WEEKS_PER_MONTH = 4.345


def ga_weeks_to_months(ga_weeks: float) -> float:
    """Convertit un âge gestationnel (semaines) en mois postnatals (terme = 0)."""
    return (float(ga_weeks) - 40.0) / _WEEKS_PER_MONTH


def months_to_ga_weeks(age_months: float) -> float:
    """Convertit des mois postnatals en âge gestationnel équivalent (semaines)."""
    return 40.0 + float(age_months) * _WEEKS_PER_MONTH


# --- Bande fœtale (volume cérébral total ±2σ, cm³) par âge gestationnel -------
# Centre ≈ midpoint ; bande ~±25 %. Handoff lisse vers la table postnatale au
# terme (40 SG ≈ (350, 525) vs postnatal mois 0 (310, 540)).
FETAL_BRAIN_VOLUME_CM3: dict[int, tuple[float, float]] = {
    18: (15.0, 32.0),
    20: (26.0, 52.0),
    22: (42.0, 78.0),
    24: (62.0, 112.0),
    26: (88.0, 152.0),
    28: (118.0, 198.0),
    30: (152.0, 250.0),
    32: (188.0, 300.0),
    34: (228.0, 358.0),
    36: (268.0, 412.0),
    38: (310.0, 470.0),
    40: (350.0, 525.0),
}

# --- Bande postnatale (miroir de adaptive_mask_fusion._NORMATIVE_BRAIN_VOLUME_CM3)
# Un test asserte l'égalité quand le module amont est importable (anti-dérive).
POSTNATAL_BRAIN_VOLUME_CM3: dict[int, tuple[float, float]] = {
    0:  (310.0, 540.0),
    1:  (355.0, 590.0),
    2:  (405.0, 640.0),
    3:  (455.0, 665.0),
    4:  (505.0, 715.0),
    5:  (550.0, 760.0),
    6:  (585.0, 960.0),
    7:  (625.0, 1010.0),
    8:  (660.0, 1050.0),
    9:  (705.0, 1100.0),
    12: (770.0, 1180.0),
    15: (835.0, 1250.0),
    18: (875.0, 1300.0),
    24: (940.0, 1400.0),
    36: (1050.0, 1400.0),
    48: (1110.0, 1460.0),
    60: (1150.0, 1490.0),
}

# Facteurs de bornes (identiques à PATCH P12 pour cohérence avec l'aval).
_DEFAULT_BOUND_FACTORS = {
    "min": 0.70,
    "soft_min": 0.90,
    "soft_max": 1.10,
    "max": 1.35,
}


def _unified_anchors() -> list[tuple[float, float, float]]:
    """Liste triée (age_months, nmin, nmax) fusionnant fœtal + postnatal.

    Les ancres fœtales sont converties en mois (négatifs). En cas de
    chevauchement au terme, la table postnatale (mois ≥ 0) prime.
    """
    anchors: dict[float, tuple[float, float]] = {}
    for ga, band in FETAL_BRAIN_VOLUME_CM3.items():
        anchors[round(ga_weeks_to_months(ga), 4)] = band
    # postnatal prime au terme : écrase tout doublon proche de 0
    for month, band in POSTNATAL_BRAIN_VOLUME_CM3.items():
        anchors[float(month)] = band
    return sorted((m, lo, hi) for m, (lo, hi) in anchors.items())


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Interpolation linéaire 1D avec clamp aux bornes (xs trié croissant)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def expected_volume_band(age_months: float) -> tuple[float, float]:
    """Bande normative (nmin, nmax) cm³ à un âge donné (mois ; négatif = fœtal)."""
    anchors = _unified_anchors()
    xs = [a[0] for a in anchors]
    return (
        _interp(float(age_months), xs, [a[1] for a in anchors]),
        _interp(float(age_months), xs, [a[2] for a in anchors]),
    )


def expected_volume_cm3(age_months: float) -> float:
    """Volume cérébral attendu (midpoint de la bande) à un âge donné."""
    lo, hi = expected_volume_band(age_months)
    return 0.5 * (lo + hi)


def expected_volume_band_ga(ga_weeks: float) -> tuple[float, float]:
    """Bande normative par âge gestationnel (semaines)."""
    return expected_volume_band(ga_weeks_to_months(ga_weeks))


def estimate_age_months_from_volume(volume_cm3: float) -> float:
    """Estime l'âge (mois postnatals ; négatif = fœtal) depuis un volume cérébral.

    Inverse monotone de la courbe midpoint. Robuste car la croissance est
    strictement croissante. Sert à choisir le bon bin d'atlas et les bonnes
    bornes volumétriques **sans dépendre de l'alignement atlas**.
    """
    anchors = _unified_anchors()
    ages = [a[0] for a in anchors]
    mids = [0.5 * (a[1] + a[2]) for a in anchors]
    # mids est croissant → on inverse en interpolant age = f(volume)
    return _interp(float(volume_cm3), mids, ages)


def estimate_ga_weeks_from_volume(volume_cm3: float) -> float:
    """Estime l'âge gestationnel (semaines) depuis un volume cérébral."""
    return months_to_ga_weeks(estimate_age_months_from_volume(volume_cm3))


def volume_bounds(
    age_months: float,
    factors: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Bornes volumétriques (min/soft_min/soft_max/max) cm³ pour un âge.

    Format directement consommable par la config d'``adaptive_fusion`` du runner
    (clés ``min_volume_cm3``, ``soft_min_volume_cm3``, ``soft_max_volume_cm3``,
    ``max_volume_cm3``). Facteurs par défaut = PATCH P12.
    """
    f = dict(_DEFAULT_BOUND_FACTORS)
    if factors:
        f.update(factors)
    nmin, nmax = expected_volume_band(age_months)
    return {
        "min_volume_cm3": nmin * f["min"],
        "soft_min_volume_cm3": nmin * f["soft_min"],
        "soft_max_volume_cm3": nmax * f["soft_max"],
        "max_volume_cm3": nmax * f["max"],
    }


def volume_bounds_ga(ga_weeks: float, factors: Mapping[str, float] | None = None) -> dict[str, float]:
    """Bornes volumétriques par âge gestationnel (semaines)."""
    return volume_bounds(ga_weeks_to_months(ga_weeks), factors)


def _selftest() -> None:
    """Vérifications internes (lancées via ``python -m scripts.normative_brain_growth``)."""
    # 1. Conversion GA ↔ mois cohérente.
    assert abs(ga_weeks_to_months(40) - 0.0) < 1e-9
    assert abs(months_to_ga_weeks(0) - 40.0) < 1e-9
    assert ga_weeks_to_months(28) < 0  # fœtal = mois négatif

    # 2. Monotonie de la croissance (midpoint strictement croissant).
    anchors = _unified_anchors()
    mids = [0.5 * (a[1] + a[2]) for a in anchors]
    assert all(mids[i] < mids[i + 1] for i in range(len(mids) - 1)), "croissance non monotone"

    # 3. Inverse cohérent avec direct (round-trip à ±1 mois).
    for age in (-5.0, -3.0, 0.0, 3.0, 6.0, 12.0):
        v = expected_volume_cm3(age)
        back = estimate_age_months_from_volume(v)
        assert abs(back - age) < 1.0, f"round-trip âge {age} → {back}"

    # 4. Plage fœtale réaliste vs FetalFMRI (~40-570 cm³).
    lo28, hi28 = expected_volume_band_ga(28)
    assert 100 < lo28 < hi28 < 250, (lo28, hi28)
    v_term = expected_volume_cm3(0.0)
    assert 380 < v_term < 480, v_term

    # 5. Estimation GA depuis volume fœtal typique.
    ga = estimate_ga_weeks_from_volume(150.0)
    assert 26 < ga < 32, ga  # ~150 cm³ ≈ 28-30 SG

    # 6. Bornes ordonnées.
    b = volume_bounds(0.0)
    assert b["min_volume_cm3"] < b["soft_min_volume_cm3"] < b["soft_max_volume_cm3"] < b["max_volume_cm3"]

    # 7. Anti-dérive : si la table amont est importable, elle doit coïncider.
    try:
        from .adaptive_mask_fusion import _NORMATIVE_BRAIN_VOLUME_CM3 as _UP  # type: ignore
    except Exception:
        try:
            from adaptive_mask_fusion import _NORMATIVE_BRAIN_VOLUME_CM3 as _UP  # type: ignore
        except Exception:
            _UP = None
    if _UP is not None:
        assert dict(_UP) == POSTNATAL_BRAIN_VOLUME_CM3, "dérive vs _NORMATIVE_BRAIN_VOLUME_CM3"

    print("normative_brain_growth: tous les self-tests OK")
    print(f"  term (40 SG / 0 mo): {v_term:.0f} cm³  | 28 SG: {expected_volume_cm3(ga_weeks_to_months(28)):.0f} cm³")
    print(f"  inverse 150 cm³ → {ga:.1f} SG ; 450 cm³ → {estimate_age_months_from_volume(450):.1f} mo")


if __name__ == "__main__":
    _selftest()
