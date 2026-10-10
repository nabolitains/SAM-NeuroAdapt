# `dataset/` — atlas library used at run time

This folder is **empty in the repository**: no atlas template is redistributed.

```
dataset/
├── atlas_library/          ← runtime atlas library read by the pipeline   (see atlas_library/README.md)
└── raw_external_atlases/   ← archives you download manually              (see raw_external_atlases/README.md)
```

Both are filled by you and by `scripts/prepare_external_atlases.py`. The pipeline reads the
library from `dataset/atlas_library/` unless the environment variable
`SAM_NEUROADAPT_ATLAS_ROOT` points elsewhere.
