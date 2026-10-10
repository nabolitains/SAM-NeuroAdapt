# `fetal/` — optional fetal templates (FetalFMRI only)

Optional. Without fetal templates, FetalFMRI is processed with neonatal templates as proxies,
which is the configuration reported in the article. Fetal atlases are distributed under
registration or data-use agreements and must be downloaded manually.

## Where to place them

`dataset/raw_external_atlases/fetal/<source>/`, with `<source>` one of `serag`, `crl`,
`dhcp_fetal`, `dhcp_fmri`.

For each gestational age, provide a **pair** of NIfTI files — a template and a brain mask —
whose names contain the gestational age (for example `GA28`, `STA28` or `28w`). The mask
file name must contain `mask`, `brain_mask`, `brainmask`, `_seg` or `tissue`; the other file
of the pair is taken as the template. Example:

```
dataset/raw_external_atlases/fetal/serag/
├── fetal_STA28_T2.nii.gz
├── fetal_STA28_brain_mask.nii.gz
└── …
```

## Building the fetal bank

```bash
python -c "from pathlib import Path; from scripts.prepare_external_atlases import prepare_fetal_manual_bank as P; P(Path('dataset/raw_external_atlases'), Path('dataset'), source='serag')"
```

This writes `dataset/atlas_library/fetal_<source>_ga<WW>/`, which the FetalFMRI routing then
uses instead of the neonatal proxies.
