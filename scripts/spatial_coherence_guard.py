"""Garde de cohérence spatiale du masque cérébral (post-refinement adulte).

Diagnostic multi-seed IBSR18 (2026-07-24) : sur config identique et graine
identique, certains étages de raffinement adulte (boundary-refine, ibsr-annotation
-align) TÉLÉPORTENT spatialement le masque de ~30 voxels en Z (recouvrement
IoU≈0 avec leur propre entrée), produisant un Dice final de 0.00. Le phénomène
est non-déterministe (déclenché par la variation run-to-run du masque SAM2 amont)
et n'est PAS attrapé par `inflation_guard` (qui ne surveille que l'inflation de
volume, pas le déplacement spatial).

Distinction nette mesurée sur IBSR18 (N=20) :
  - téléportations (Mode A)          : IoU(sortie, entrée) = 0.000
  - transitions saines / raffinements: IoU(sortie, entrée) ≥ 0.562
Un seuil τ=0.40 revert EXACTEMENT les téléportations sans jamais toucher un
raffinement légitime (marge des deux côtés). En particulier, les remplacements
de l'adjudicateur (IoU 0.56–0.65, conservés par choix) passent la garde.

Principe : un raffinement qui déplace le masque hors de lui-même (recouvrement
quasi nul avec son entrée) a catastrophiquement échoué ; on revient à l'entrée.
Aucune vérité terrain, aucune dépendance GPU, déterministe. Opt-in (défaut OFF).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import nibabel as nib


DEFAULT_IOU_THRESHOLD = 0.40
DEFAULT_MIN_BACKUP_VOXELS = 1000  # ne pas comparer à un backup dégénéré/vide


def _load_bool(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj) > 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float((a & b).sum()) / union


def apply_spatial_coherence_guard(
    mask_path: str | Path,
    backup_path: Optional[str | Path],
    stage: str,
    input_image_path: Optional[str | Path] = None,
    brain_path: Optional[str | Path] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Revert le masque courant à son entrée (backup) s'il a été téléporté.

    - `mask_path`   : masque produit par l'étage `stage` (écrasé si revert).
    - `backup_path` : snapshot AVANT l'étage (`_pre-<stage>`), = l'entrée de l'étage.
    - `stage`       : nom de l'étage (pour le préfixe des métriques).
    - `input_image_path`/`brain_path` : pour régénérer le brain si revert (optionnel).
    Retourne un dict de métriques préfixé `spatial_guard_<stage>_`.
    """
    cfg = dict(config or {})
    pfx = f"spatial_guard_{stage}_"
    out: dict[str, Any] = {
        f"{pfx}enabled": bool(cfg.get("enabled", False)),
        f"{pfx}reverted": False,
        f"{pfx}reason": "disabled",
    }
    if not cfg.get("enabled", False):
        return out
    mask_path = Path(mask_path)
    if backup_path is None:
        out[f"{pfx}reason"] = "no_backup"
        return out
    backup_path = Path(backup_path)
    if not mask_path.exists() or not backup_path.exists():
        out[f"{pfx}reason"] = "missing_mask_or_backup"
        return out

    tau = float(cfg.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
    min_vox = int(cfg.get("min_backup_voxels", DEFAULT_MIN_BACKUP_VOXELS))
    cur = _load_bool(mask_path)
    bak = _load_bool(backup_path)
    bak_vox = int(bak.sum())
    out[f"{pfx}iou_threshold"] = tau
    out[f"{pfx}backup_voxels"] = bak_vox

    if bak_vox < min_vox:
        # entrée dégénérée : la garde ne peut pas juger, on ne touche à rien
        out[f"{pfx}reason"] = "degenerate_backup"
        return out

    iou = _iou(cur, bak)
    out[f"{pfx}iou_vs_input"] = round(iou, 4)
    if iou >= tau:
        out[f"{pfx}reason"] = "coherent"
        return out

    # téléportation détectée → revert à l'entrée
    shutil.copyfile(backup_path, mask_path)
    out[f"{pfx}reverted"] = True
    out[f"{pfx}reason"] = "teleport_reverted"
    if input_image_path is not None and brain_path is not None:
        try:
            from .neuro_utils import write_brain_from_mask
            write_brain_from_mask(Path(input_image_path), mask_path, Path(brain_path))
        except Exception as exc:  # pragma: no cover - I/O best effort
            out[f"{pfx}brain_rewrite_error"] = str(exc)
    return out


def _selftest() -> None:
    """Auto-test minimal sans I/O disque : la logique IoU/seuil."""
    a = np.zeros((10, 10, 10), bool); a[2:6, 2:6, 2:6] = True
    b = a.copy()
    assert _iou(a, b) == 1.0
    c = np.zeros_like(a); c[2:6, 2:6, 6:10] = True  # décalé, recouvrement nul
    assert _iou(a, c) == 0.0
    # léger raffinement : haute IoU
    d = a.copy(); d[2, 2, 2] = False
    assert _iou(a, d) > 0.95
    print("spatial_coherence_guard selftest OK")


if __name__ == "__main__":
    _selftest()
