# `resources/` — comparison methods (only needed to reproduce the baselines)

The SAM-NeuroAdapt pipeline itself needs nothing in this folder. It is used only by the
baseline runners (`scripts/m2_external_sota_runner.py`, `scripts/m2_pediatric_sota_runner.py`),
which re-run the comparison methods of the article. **Nothing is redistributed here.**

| Method | How the code finds it | What to install |
|---|---|---|
| **HD-BET** | `hd-bet` on the PATH or in `.venv/bin/` | installed by `requirements-sam-neuroadapt.txt` (`HD-BET==2.0.1`); its weights are downloaded by HD-BET itself |
| **SynthStrip** | `resources/tools/mri_synthstrip`, else `mri_synthstrip` on the PATH, or the variables below | FreeSurfer's `mri_synthstrip`, from the [SynthStrip page](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) |
| **FSL BET** (supplementary only) | `bet` on the PATH, `$FSLDIR/bin/bet`, or `FSL_BET_BIN` | [FSL](https://fsl.fmrib.ox.ac.uk/fsl/) |

## Expected layout for SynthStrip (exact names)

```
resources/
├── tools/
│   └── mri_synthstrip           ← executable (or put mri_synthstrip on the PATH)
└── models/
    └── synthstrip.1.pt          ← SynthStrip weights, if your mri_synthstrip needs them
```

The locations can be overridden with the environment variables `SYNTHSTRIP_BIN`,
`SYNTHSTRIP_PY` and `SYNTHSTRIP_WT` (weights). These tools are distributed by their
authors under their own licences.
