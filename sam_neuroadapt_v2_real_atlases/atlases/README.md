# `atlases/` — MNI152NLin2009cAsym base template

This folder is **empty in the repository**. It must contain the 2 mm MNI152NLin2009cAsym
template and its brain mask from [TemplateFlow](https://www.templateflow.org/)
(template `tpl-MNI152NLin2009cAsym`), under their original names:

```
sam_neuroadapt_v2_real_atlases/atlases/
├── tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz
└── tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz
```

`scripts/prepare_external_atlases.py` copies these two files into
`dataset/atlas_library/mni152_nlin_2009c/` (adult atlas library); see
`dataset/atlas_library/README.md`.
