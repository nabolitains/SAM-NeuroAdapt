# `dataset/atlas_library/` — the external atlas library

This folder is **empty in the repository**. The templates are public but each is
distributed by its authors under its own licence, so **you download them** and
`scripts/prepare_external_atlases.py` builds this library. No subject from an evaluation
dataset is ever used as an atlas.

## What the code expects in every atlas folder

The pipeline looks up atlases **by folder name** (the names below are fixed in
`scripts/external_atlas_config.py`). Each folder must contain at least:

```
dataset/atlas_library/<atlas_name>/
├── MNI152_T1_2mm.nii.gz    ← the template image (historical alias name, whatever the template)
└── brain_mask.nii.gz       ← its binary brain mask
```

`prepare_external_atlases.py` writes these two files, plus the native-resolution copies
(`<atlas_name>_T1w.nii.gz`, `_T2w`, `_label_dseg` when available) and an
`atlas_manifest.json` recording the source and the validation of each file. A missing atlas
is skipped silently, so check the summary JSON the script writes in this folder.

## Atlases and how to obtain them

**A — downloaded automatically** by `python -m scripts.prepare_external_atlases` (raw files
go to `data/external_atlases/`):

| Folder name(s) | Template | Used for |
|---|---|---|
| `oasis30ants` | TemplateFlow `tpl-OASIS30ANTs` | adults (IBSR-20, Mindboggle-101) |
| `mniinfant_cohort_01` … `mniinfant_cohort_11` | TemplateFlow `tpl-MNIInfant` | infants (BOBs, SynthStripInfant) |
| `unc_bcp_06m` | UNC-BCP 4D infant atlas v2 (NITRC) | neonatal / fetal fall-back |

**B — from files you place yourself**, then the same command:

| Folder name(s) | Template | What to place, and where | Used for |
|---|---|---|---|
| `mni152_nlin_2009c` | TemplateFlow `tpl-MNI152NLin2009cAsym` | the two files listed in `sam_neuroadapt_v2_real_atlases/atlases/README.md` | adults |
| `unc_infant_neo`, `unc_infant_1yr`, `unc_infant_2yr` | UNC Infant 0-1-2 atlases (Shi et al., 2011) | `data/UNCInfant012Atlases/UNCInfant012Atlases*.zip` | infants (BOBs) |
| `fbn125_neonatal` | FinnBrain FBN-125 ([NITRC](https://www.nitrc.org/projects/finnbrain/)) | `dataset/raw_external_atlases/fbn125_neonatal/FBN-125-template-et-atlases.zip` | neonates, fetal fall-back |
| `alberts_gousias_01` … `_20` | ALBERTs neonatal atlases (Gousias et al.) | `dataset/raw_external_atlases/alberts_gousias/` with `ALBERTs-Gousias-neonatal-atlases-images.tar.gz` and `ALBERTs-Gousias-neonatal-atlases-segmentations.zip` | neonates (M-CRIB) |

**C — adult templates**, prepared with `python -m scripts.prepare_external_atlases --adult-templates`:

| Folder name | Template | Obtained | Used for |
|---|---|---|---|
| `sri24` | SRI24 (Rohlfing et al., 2010) | downloaded automatically from NITRC, or place `sri24_anatomy_nifti.zip` and `sri24_anatomy_unstripped_nifti.zip` in `dataset/raw_external_atlases/sri24/` | Mindboggle-101 |
| `colin27_2008` | Colin27 2008 (MNI) | downloaded automatically, or place `mni_colin27_2008_nifti.zip` in `dataset/raw_external_atlases/colin27_2008/` | Mindboggle-101 |
| `icbm452` | ICBM452 ([LONI](https://resource.loni.usc.edu/resources/atlases/)) | place `ICBM_452_T1_Atlas_Affine_Transformations.zip` in `dataset/raw_external_atlases/icbm452/` | Mindboggle-101 |
| `lpba40_average` | LPBA40 average ([LONI](https://resource.loni.usc.edu/resources/atlases/)) | place `LPBA40_AIR_Atlas_NIFTI_format.zip` (or `LBPA40_Atlas_MBAT_v3.zip`) in `dataset/raw_external_atlases/lpba40_average/` | Mindboggle-101 |
| `miitra_older_adult` | MIITRA ([NITRC](https://www.nitrc.org/projects/miitra/)) | place `MIITRA-T1w-1mm.nii.gz` and `MIITRA-BrainMask-05mm.nii.gz` in `dataset/raw_external_atlases/MIITRA/` | Mindboggle-101, collapse-rescue |

`oasis30ants` and `mni152_nlin_2009c` (A and B above) complete the adult set.

**D — only for the supplementary datasets**, built with the functions of the same script:

```bash
# UNC-BCP monthly templates unc_bcp_01m … unc_bcp_09m (SynthStripInfant)
python -c "from pathlib import Path; from scripts.prepare_external_atlases import prepare_unc_bcp_4d_months as f; f(Path('data/external_atlases'), Path('dataset'), range(1, 10))"
# dHCP neonatal templates dhcp_neonatal_ga36 … ga44 (M-CRIB)
python -c "from pathlib import Path; from scripts.prepare_external_atlases import prepare_dhcp_neonatal as f; f(Path('data/external_atlases'), Path('dataset'))"
```

Fetal templates (`fetal_*`, optional, FetalFMRI only) are described in
`dataset/raw_external_atlases/fetal/README.md`; without them the neonatal templates above
are used as proxies.

## Minimum set per benchmark

| To reproduce | Atlas folders required |
|---|---|
| BOBs | `mniinfant_cohort_01` … `_05`, `unc_infant_neo`, `unc_infant_1yr` |
| IBSR-20 | `oasis30ants`, `mni152_nlin_2009c` |
| Mindboggle-101 | the seven adult folders of tables A–C |
| SynthStripInfant | `mniinfant_cohort_01/03/05/07/09`, `unc_bcp_01m/03m/05m/07m/09m`, `unc_infant_neo` |
| FetalFMRI | `fetal_*` if available, otherwise `fbn125_neonatal`, `unc_infant_neo`, `unc_bcp_06m` |
