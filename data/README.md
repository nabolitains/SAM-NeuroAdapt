# `data/` — raw datasets (you download them; nothing is redistributed)

This folder is **empty in the repository**. Each evaluation dataset must be downloaded by
you from its original provider, under its own licence and data-use terms, and placed here
**with the folder and file names below**: `scripts/standardize_datasets.py` looks for these
exact paths. It then writes the standardized copies and the subject manifests to
`datasets/` (see `datasets/README.md`).

## Primary benchmarks of the article

| Dataset | Source | Expected location and names |
|---|---|---|
| **BOBs** (Baby Open Brains; infants, 71 visits) | OpenNeuro `ds005450` (Feczko et al., 2025) | `data/BOBs/ds005450/sub-*/ses-*/anat/` containing `*_T1w.nii.gz`, `*_T2w.nii.gz`, `*desc-aseg_brainmask.nii.gz` (and optionally `*desc-aseg_dseg.nii.gz`) |
| **IBSR-20** (adults, n = 20) — `IBSR18` in the code | [IBSR on NITRC](https://www.nitrc.org/projects/ibsr), "20 Normal Subjects" collection | `data/IBSR/20Normals_T1.tgz` and `data/IBSR/20Normals_T1_brain.tgz` (the two archives, not extracted) |
| **Mindboggle-101** (adults, n = 101) | Mindboggle-101, Harvard Dataverse (Klein & Tourville, 2012) | the archive `dataverse_files.zip` — **edit the `ZIP = Path(...)` line at the top of `scripts/prepare_mindboggle101.py`** to point to it, then run that script |

## Supplementary datasets (Supplementary Section S1)

| Dataset | Source | Expected location and names |
|---|---|---|
| **SynthStripInfant** (16 infants) | [SynthStrip evaluation data v1.5](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) | `data/SynthStrip/synthstrip_data_v1.5/<subject>/image.nii.gz` and `mask.nii.gz`; the infant subset is selected automatically |
| **FetalFMRI** (3 fetuses) | [OpenNeuro `ds003090`](https://openneuro.org/datasets/ds003090) | `data/FetalFMRI/ds003090/sub-*/ses-*/func/*_bold.nii.gz` with the manual masks under `data/FetalFMRI/ds003090/derivatives/manual-masks/` (same relative paths) |
| **NFBS** | Preprocessed Connectomes Project (Puccio et al., 2016) | `data/NFBS/NFBS_Dataset/NFBS_Dataset/<subject>/` containing `*_T1w.nii.gz` and `*_brainmask.nii.gz` |
| **LPBA40** | [LONI atlases](https://resource.loni.usc.edu/resources/atlases/) (Shattuck et al., 2008) | anywhere under `data/LPBA40/`: `<id>/<id>.native.mri.hdr/.img` and `<id>/<id>.native.brain.mask.hdr/.img` |
| **M-CRIB** family | M-CRIB atlas (Alexander et al., 2017) | `data/MCRIB/` (or `data/M-CRIB/`) and `data/MCRIB_WM/` with the `P<NN>` T1/T2 images and label maps |
| **iSeg-2017** | iSeg-2017 challenge (Wang et al., 2019) | `data/iSeg-2017/iSeg-2017.zip` |

## Raw atlas downloads

Two sub-folders of `data/` receive atlas material (see `dataset/atlas_library/README.md`):

* `data/external_atlases/` — filled **automatically** by `scripts/prepare_external_atlases.py`
  (TemplateFlow, UNC-BCP, dHCP, SRI24, Colin27 downloads). Do not fill by hand.
* `data/UNCInfant012Atlases/` — place here the **UNC Infant 0-1-2 atlases** archive
  (Shi et al., 2011), named `UNCInfant012Atlases*.zip`; it must contain
  `infant-neo-withSkull.nii.gz` / `infant-neo-seg.nii.gz` and their 1-year and 2-year
  counterparts. Needed for BOBs.

Only the datasets you intend to reproduce are needed; the standardizer skips missing ones
with a warning.
