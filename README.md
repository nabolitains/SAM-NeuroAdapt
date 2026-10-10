# SAM-NeuroAdapt

**Atlas-guided, training-free brain extraction (skull-stripping) for infant and adult MRI.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23287123.svg)](https://doi.org/10.5281/zenodo.23287123)

Source code accompanying the manuscript *"SAM-NeuroAdapt: an atlas-guided, training-free
pipeline for skull-stripping in infant and adult MRI"* (submitted to *Brain Informatics*).

---

## Overview

SAM-NeuroAdapt performs brain extraction **without any task-specific model training**:

1. **Atlas selection** — a template is chosen from a library of 21 external, publicly
   available templates spanning the lifespan. When the subject's age is recorded, age drives
   the choice; otherwise image–atlas alignment does, with no age weighting.
2. **Registration** — the template is registered to the subject (N4 bias-field correction,
   rigid/affine alignment, then ANTs SyN).
3. **Prompted segmentation** — the registered atlas supplies point and box prompts to a
   **frozen** Segment Anything Model 2 (SAM2); its weights are never updated.
4. **Refinement** — the mask is refined by classical intensity-based methods (Chan–Vese
   active contours, random-walker candidates for infants), constrained by deterministic checks.

No reference mask is used at inference, and no subject from an evaluation dataset is used
as an atlas.

## Main results reported in the article

Mean Dice and average symmetric surface distance (ASSD, mm), compared with HD-BET and
SynthStrip. *p*-values are Benjamini–Hochberg-adjusted across nine comparisons; see the
article for the complete statistics.

| Benchmark | SAM-NeuroAdapt | HD-BET | SynthStrip |
|---|---|---|---|
| BOBs, infants (71 visits, 51 participants) — Dice | **0.920** | 0.905 | 0.903 |
| BOBs — ASSD (mm) | **2.07** | 5.11 | 4.96 |
| IBSR-20, adults (n = 20) — Dice¹ | 0.939 | 0.922 | 0.902 |
| IBSR-20 — ASSD (mm)¹ | **3.13** | 3.92 | 4.99 |
| Mindboggle-101, adults (n = 101) — Dice | 0.942 | 0.943 | 0.881 |
| Mindboggle-101 — ASSD (mm) | **2.42** | 3.21 | 5.91 |

Bold: significantly better than HD-BET after adjustment. ¹ Mean over four uninterrupted
runs; the IBSR-20 Dice difference is not significant after adjustment, and the ASSD
difference is significant under Benjamini–Hochberg but not under Holm.

A stage-wise ablation shows that the age-matched atlas alone reaches the infant Dice of the
complete pipeline (0.923 vs 0.920), while the later stages lower surface distances. Without
atlas guidance, SAM2 fails (Dice < 0.80) on 26–100 % of cases depending on the population.

## What this repository does **not** contain

No imaging data, subject list, atlas template or model weight is redistributed here: each
is distributed by its authors under its own licence. The repository ships the **empty
folders the code expects**, each with a `README.md` that says what to download, where to
place it and under which exact name:

| Folder | Contents you provide | Guide |
|---|---|---|
| `sam_neuroadapt_v2_real_atlases/models/` | SAM2.1 Hiera-Large checkpoint and configuration | [README](sam_neuroadapt_v2_real_atlases/models/README.md) |
| `sam_neuroadapt_v2_real_atlases/atlases/` | MNI152NLin2009cAsym base template (TemplateFlow) | [README](sam_neuroadapt_v2_real_atlases/atlases/README.md) |
| `dataset/atlas_library/` | the external atlas library, built by a script | [README](dataset/atlas_library/README.md) |
| `dataset/raw_external_atlases/` | atlas archives that must be downloaded by hand | [README](dataset/raw_external_atlases/README.md) |
| `data/` | the raw evaluation datasets | [README](data/README.md) |
| `datasets/` | nothing: written by the preparation scripts | [README](datasets/README.md) |
| `resources/` | comparison methods (HD-BET, SynthStrip, FSL BET), baselines only | [README](resources/README.md) |

## Installation

Tested with Python 3.12, PyTorch 2 with CUDA, on a single 4 GB GPU. Run every command from
the repository root.

```bash
git clone https://github.com/nabolitains/SAM-NeuroAdapt.git && cd SAM-NeuroAdapt
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-sam-neuroadapt.txt
```

## Preparing the model, atlases and data

```bash
# 1. SAM2 checkpoint and configuration -> sam_neuroadapt_v2_real_atlases/models/  (manual)
# 2. Base template                     -> sam_neuroadapt_v2_real_atlases/atlases/ (manual)
# 3. Atlas library (downloads the open templates, uses the archives you placed by hand)
python -m scripts.prepare_external_atlases
python -m scripts.prepare_external_atlases --adult-templates
# 4. Datasets: raw downloads in data/ -> standardized copies and manifests in datasets/
python -m scripts.standardize_datasets --datasets BOBs,IBSR18,SynthStrip,FetalFMRI
python -m scripts.prepare_mindboggle101   # edit the ZIP path at the top of the script first
```

In the code and file names, the adult IBSR benchmark is labelled `IBSR18`; it corresponds to
the IBSR "20 Normal Subjects" collection (n = 20) called IBSR-20 in the article.

## Running the pipeline

The pipeline core (preprocessing, registration and the SAM2 extractor) is defined in
`sam_neuroadapt_pipeline.ipynb`; the command-line launchers load it automatically.

```bash
# Infants (BOBs)
python -m scripts.run_m1_pediatric_isegstyle --dataset BOBs

# Adults (IBSR-20, labelled IBSR18 in the code)
SAM_NEUROADAPT_INFLATION_GUARD=1 SAM_NEUROADAPT_SPATIAL_GUARD=1 \
  python -m scripts.run_m1_adult --dataset IBSR18 --seed 20260518

# Ablation and multi-seed campaign used in the article (resumable)
python -m scripts.ablation_campaign            # run or resume
python -m scripts.ablation_campaign --status   # progress only
```

The exact configuration of each published experiment, including its environment flags, is
recorded in `scripts/ablation_campaign.py`.

| Environment flag | Effect (all default off) |
|---|---|
| `SAM_NEUROADAPT_INFLATION_GUARD=1` | reverts a refinement that grows the mask by more than 15 % |
| `SAM_NEUROADAPT_SPATIAL_GUARD=1` | reverts a stage whose output no longer overlaps its input |
| `SAM_NEUROADAPT_INTENSITY_REFINE=1` | enables the adult intensity refinement |
| `SAM_NEUROADAPT_NO_ATLAS=1` | ablation: disables atlas guidance at the SAM2 stage |

Statistics, tables and figures of the article are produced by `scripts/recompute_surface_metrics.py`,
`scripts/make_multiplicity_correction.py`, `scripts/make_effectsize_table.py`, `scripts/make_table2_*.py`
and `scripts/make_fig*.py`, from the result files written by the runs above.

## Reproducibility notes

**Run each configuration in a single uninterrupted session.** Resuming an interrupted run
can silently disable a safety check and degrade results without any error.
[`REPRODUCIBILITY_KNOWN_ISSUES.md`](REPRODUCIBILITY_KNOWN_ISSUES.md) documents this and the
other known issues, with detection snippets.

## Repository layout

| Path | Contents |
|---|---|
| `sam_neuroadapt_pipeline.ipynb` | pipeline core: preprocessing, registration, SAM2 extractor |
| `scripts/` | launchers, atlas selection, refinement, guards, evaluation, statistics and figures |
| `REPRODUCIBILITY_KNOWN_ISSUES.md` | known failure modes and how to detect them |
| `requirements-sam-neuroadapt.txt` | Python dependencies |
| `data/`, `datasets/`, `dataset/`, `sam_neuroadapt_v2_real_atlases/`, `resources/` | empty folders with a README each (see above) |

## Citation

If you use this code, please cite the accompanying article and the archived software
(see `CITATION.cff`). Version used in the article: release v1.0.0,
[doi:10.5281/zenodo.23287124](https://doi.org/10.5281/zenodo.23287124); all versions:
[doi:10.5281/zenodo.23287123](https://doi.org/10.5281/zenodo.23287123).

## Licence

MIT (see `LICENSE`). SAM2, the atlas templates and the evaluation datasets remain under
their own licences and are not redistributed here.
