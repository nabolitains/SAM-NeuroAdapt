# `dataset/raw_external_atlases/` — atlas archives you download manually

This folder is **empty in the repository**. Some templates cannot be downloaded
automatically (registration form, licence acceptance or login). Download them yourself from
the pages given in `dataset/atlas_library/README.md` and place the archives here, **without
renaming them**, in a sub-folder named after the atlas:

```
dataset/raw_external_atlases/
├── fbn125_neonatal/   FBN-125-template-et-atlases.zip
├── alberts_gousias/   ALBERTs-Gousias-neonatal-atlases-images.tar.gz
│                      ALBERTs-Gousias-neonatal-atlases-segmentations.zip
├── icbm452/           ICBM_452_T1_Atlas_Affine_Transformations.zip
├── lpba40_average/    LPBA40_AIR_Atlas_NIFTI_format.zip
├── MIITRA/            MIITRA-T1w-1mm.nii.gz, MIITRA-BrainMask-05mm.nii.gz
├── sri24/             (optional: downloaded automatically otherwise)
├── colin27_2008/      (optional: downloaded automatically otherwise)
└── fetal/             (optional, FetalFMRI only — see fetal/README.md)
```

Then run `python -m scripts.prepare_external_atlases` (and `--adult-templates` for the adult
set). Only the atlases needed for the datasets you reproduce are required.
