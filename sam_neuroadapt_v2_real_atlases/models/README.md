# `models/` — Segment Anything Model 2 (SAM2.1 Hiera-Large)

SAM-NeuroAdapt uses SAM2 **frozen**: no weight is trained or modified. The weights belong
to Meta and are **not redistributed** here; download them from the official repository,
[facebookresearch/sam2](https://github.com/facebookresearch/sam2), under its licence.

## Expected content (exact file names)

```
sam_neuroadapt_v2_real_atlases/models/
├── sam2.1_hiera_large.pt     ← checkpoint "SAM 2.1 Hiera Large"
└── sam2.1_hiera_l.yaml       ← its configuration file (`sam2/configs/sam2.1/sam2.1_hiera_l.yaml` in the SAM2 repository)
```

The code reads these two paths literally (`MODELS_DIR / "sam2.1_hiera_large.pt"` and
`MODELS_DIR / "sam2.1_hiera_l.yaml"`); other names or locations are not found.

## How to obtain them

1. Download the **SAM 2.1 Hiera Large** checkpoint linked from the "Model checkpoints"
   section of the SAM2 README, and save it here as `sam2.1_hiera_large.pt`.
2. Copy `sam2/configs/sam2.1/sam2.1_hiera_l.yaml` from the same repository and save it here
   as `sam2.1_hiera_l.yaml`.
3. The `sam2` Python package itself is installed by `requirements-sam-neuroadapt.txt`.

The checkpoint is about 900 MB and runs on a 4 GB GPU in the configuration used in the article.
