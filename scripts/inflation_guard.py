"""Garde anti-inflation du masque cérébral (post-refinement adulte).

Diagnostic IBSR18 déterministe (2026-06-24) : sur certains sujets, la chaîne de
refinement adulte (fusion adaptive + strong refine + candidate adjudicator) suit
un atlas recalé SUR-DIMENSIONNÉ et gonfle le masque dans le cou / la base du crâne
(faux positifs inférieurs massifs : +300–430 cm³, pred/GT ≈ 1.40). Le masque
SAM2+fusion de base (snapshot `pre-adaptive-fusion`), lui, était déjà bon
(Dice ≈ 0.92–0.93, niveau HD-BET). La chaîne avait donc la bonne réponse et l'a
jetée pour l'atlas oversize.

Cette garde est volontairement ASYMÉTRIQUE et conforme à la thèse « ne jamais
forcer un atlas mal dimensionné » : on n'autorise QUE le retour au masque de base
quand le refinement a INFLÉ le volume au-delà d'un ratio (le rétrécissement, lui,
est toujours laissé passer car le pruning aide en général). Empiriquement sur
IBSR18 (N=20), un seuil τ∈[1.15, 1.29] revert EXACTEMENT les 2 outliers
(0.813→0.919, 0.816→0.930) sans toucher aucun sujet où le refinement a aidé
(tous ont final/base < 1.0) → mean 0.9211→0.9321 (> HD-BET 0.9217).

Opt-in (défaut OFF) ; aucune dépendance GPU ; déterministe.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import nibabel as nib


DEFAULT_RATIO_THRESHOLD = 1.15
DEFAULT_MIN_BASE_VOLUME_CM3 = 200.0  # garde-fou : ne pas comparer à un masque de base dégénéré


def _mask_volume_cm3(path: Path) -> float:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj) > 0
    vox = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    return float(data.sum()) * vox


def apply_inflation_guard(
    mask_path: str | Path,
    base_mask_path: Optional[str | Path],
    input_image_path: Optional[str | Path] = None,
    brain_path: Optional[str | Path] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Revert le masque courant au masque de base s'il a été sur-inflé.

    - `mask_path` : masque final courant (sera écrasé si revert).
    - `base_mask_path` : snapshot SAM2+fusion de base (`adaptive_fusion_backup_mask`).
    - `input_image_path` : image N4 pour régénérer le brain (optionnel).
    - `brain_path` : sortie brain à réécrire si revert (optionnel).
    Retourne un dict de métriques préfixé `inflation_guard_`.
    """
    cfg = dict(config or {})
    out: dict[str, Any] = {
        "inflation_guard_enabled": bool(cfg.get("enabled", False)),
        "inflation_guard_used": False,
        "inflation_guard_reason": "",
        "inflation_guard_ratio": None,
        "inflation_guard_volume_before_cm3": None,
        "inflation_guard_volume_after_cm3": None,
    }
    if not out["inflation_guard_enabled"]:
        out["inflation_guard_reason"] = "disabled"
        return out
    if not base_mask_path or not Path(base_mask_path).exists():
        # ⚠️ PIÈGE DE REPRODUCTIBILITÉ CONNU — voir REPRODUCIBILITY_KNOWN_ISSUES.md §1.
        # Cette garde échoue en mode OUVERT : sans masque de base, elle ne vérifie rien
        # et laisse passer le masque tel quel. C'est bénin dans un run ininterrompu,
        # mais DANGEREUX sur un run REPRIS après interruption : les snapshots `_pre-*`
        # du segment précédent existent déjà, l'étage amont ne réécrit donc pas son
        # backup (`if not backup.exists()`) et ne renvoie pas `*_backup_mask` → on
        # arrive ici → une inflation de +40-48 % (atlas oversize fuyant dans le cou /
        # la base du crâne, précisément ce que cette garde doit attraper) passe sans
        # correction et fait chuter le Dice de ~0.95 à ~0.71-0.82.
        # Mesuré (IBSR18, config identique) : graine interrompue en 3 segments =
        # 0.8886 (no_base_mask 16/20, 8/20 sujets gonflés) vs 4 graines ininterrompues
        # = 0.938-0.939 (écart-type 0.0007).
        # NON CORRIGÉ VOLONTAIREMENT : les résultats publiés ont été produits avec ce
        # comportement ; le modifier a posteriori invaliderait leur cohérence.
        # Détection a posteriori : inflation_guard_reason == "no_base_mask"
        #                          ET vol_pred_cm3 / vol_gs_cm3 > 1.15
        out["inflation_guard_reason"] = "no_base_mask"
        return out

    mask_path = Path(mask_path)
    base_mask_path = Path(base_mask_path)
    threshold = float(cfg.get("ratio_threshold", DEFAULT_RATIO_THRESHOLD))
    min_base = float(cfg.get("min_base_volume_cm3", DEFAULT_MIN_BASE_VOLUME_CM3))

    cur_vol = _mask_volume_cm3(mask_path)
    base_vol = _mask_volume_cm3(base_mask_path)
    out["inflation_guard_volume_before_cm3"] = round(cur_vol, 3)
    if base_vol < min_base:
        out["inflation_guard_reason"] = f"base_volume_too_small ({base_vol:.1f} cm3)"
        return out

    ratio = cur_vol / base_vol if base_vol > 0 else float("inf")
    out["inflation_guard_ratio"] = round(ratio, 4)
    if ratio <= threshold:
        out["inflation_guard_reason"] = f"no_inflation (ratio {ratio:.3f} <= {threshold:.3f})"
        out["inflation_guard_volume_after_cm3"] = round(cur_vol, 3)
        return out

    # Inflation pathologique → revenir au masque de base.
    shutil.copyfile(base_mask_path, mask_path)
    out["inflation_guard_used"] = True
    out["inflation_guard_reason"] = (
        f"reverted_to_base (ratio {ratio:.3f} > {threshold:.3f}; "
        f"{cur_vol:.1f} -> {base_vol:.1f} cm3)"
    )
    out["inflation_guard_volume_after_cm3"] = round(base_vol, 3)

    # Régénérer le brain masqué pour rester cohérent avec le masque reverté.
    if input_image_path and brain_path and Path(input_image_path).exists():
        img = nib.load(str(input_image_path))
        m = nib.load(str(base_mask_path))
        brain = np.asarray(img.dataobj, dtype=np.float32) * (np.asarray(m.dataobj) > 0)
        nib.save(nib.Nifti1Image(brain, img.affine, img.header), str(brain_path))

    return out


if __name__ == "__main__":
    # Self-test : masque de base plausible + masque inflé synthétique.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    aff = np.diag([1.0, 1.0, 1.0, 1.0])

    def save(name, arr):
        p = tmp / name
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), aff), str(p))
        return p

    # base = cube 40^3 (= 64 cm3) ; inflé = cube 50^3 (= 125 cm3) ratio 1.95
    base = np.zeros((80, 80, 80), np.uint8); base[20:60, 20:60, 20:60] = 1
    infl = np.zeros((80, 80, 80), np.uint8); infl[15:65, 15:65, 15:65] = 1
    # léger rétrécissement = cube 38^3, ratio 0.86 (ne doit PAS revert)
    shrunk = np.zeros((80, 80, 80), np.uint8); shrunk[21:59, 21:59, 21:59] = 1

    bp = save("base.nii.gz", base)

    # cas 1 : désactivé → no-op
    mp = save("m1.nii.gz", infl.copy())
    r = apply_inflation_guard(mp, bp, config={"enabled": False})
    assert r["inflation_guard_used"] is False and r["inflation_guard_reason"] == "disabled", r

    # cas 2 : inflation → revert
    mp = save("m2.nii.gz", infl.copy())
    r = apply_inflation_guard(mp, bp, config={"enabled": True, "ratio_threshold": 1.15, "min_base_volume_cm3": 10})
    assert r["inflation_guard_used"] is True, r
    assert abs(_mask_volume_cm3(mp) - _mask_volume_cm3(bp)) < 1e-6, "masque non reverté"
    print(f"  cas inflation : ratio={r['inflation_guard_ratio']} -> revert OK "
          f"({r['inflation_guard_volume_before_cm3']}->{r['inflation_guard_volume_after_cm3']} cm3)")

    # cas 3 : rétrécissement → PAS de revert (asymétrie)
    mp = save("m3.nii.gz", shrunk.copy())
    v_before = _mask_volume_cm3(mp)
    r = apply_inflation_guard(mp, bp, config={"enabled": True, "ratio_threshold": 1.15, "min_base_volume_cm3": 10})
    assert r["inflation_guard_used"] is False, r
    assert abs(_mask_volume_cm3(mp) - v_before) < 1e-6, "masque modifié à tort"
    print(f"  cas rétrécissement : ratio={r['inflation_guard_ratio']} -> conservé OK")

    # cas 4 : base manquante → no-op sûr
    r = apply_inflation_guard(mp, tmp / "absent.nii.gz", config={"enabled": True})
    assert r["inflation_guard_used"] is False and r["inflation_guard_reason"] == "no_base_mask", r

    print("inflation_guard self-test: OK")
