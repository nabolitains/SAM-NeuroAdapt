# Known Issues and Reproducibility Notes

This document lists failure modes we identified in this pipeline **and deliberately did
not patch**. They are documented rather than fixed because the reported results were
produced with the code as-is: silently changing guard behaviour after the fact would
invalidate the consistency of the frozen runs and risk introducing untested regressions.

Anyone reproducing or extending this work should read this file first. Each issue lists
its **mechanism**, the **evidence** we have for it, how to **detect** it in your own runs,
and how to **avoid** it.

---

## Issue #1 — Resuming an interrupted run can silently corrupt results

**Severity: HIGH.** This is the single most important caveat in this repository.

### Mechanism

Every refinement stage snapshots its input before modifying the mask, using the idiom:

```python
backup = mask_path.with_name(... "_pre-<stage>.nii.gz")
if not backup.exists():
    shutil.copyfile(mask_path, backup)
```

(see `scripts/adaptive_mask_fusion.py`, `scripts/adult_external_refinement.py`,
`scripts/iseg2017_robust_refinement.py`)

The guard `if not backup.exists()` is correct within a single run — it prevents a stage
from overwriting its own snapshot. But if a run is **interrupted** (machine reboot, power
loss, `SIGKILL`) and later **resumed**, the subject is re-processed from scratch while the
snapshot files from the *previous* segment are still on disk. The backup is therefore not
refreshed, and the stage does not report a `*_backup_mask` path for the current run.

Downstream, `apply_inflation_guard()` receives `base_mask_path=None`, records
`inflation_guard_reason = "no_base_mask"`, and **returns without checking anything**. The
guard fails *open*: when it cannot verify, it permits instead of blocking. A mask inflated
by 40–48 % (the registered atlas leaking into neck and skull base — precisely the failure
mode `inflation_guard` exists to catch) then passes through uncorrected.

### Evidence

IBSR18, five seeds, identical configuration
(`SAM_NEUROADAPT_INFLATION_GUARD=1 SAM_NEUROADAPT_SPATIAL_GUARD=1`). Only seed 20260517
was interrupted (3 execution segments across two machine reboots); seeds 20260518–21 each
ran uninterrupted:

| Seed | Segments | Dice | `no_base_mask` | Subjects >1.15×GT volume |
|---|---|---|---|---|
| 20260517 | **3** | **0.8886** | **16/20** | **8/20** |
| 20260518 | 1 | 0.9389 | 10/20 | 3/20 |
| 20260519 | 1 | 0.9380 | 10/20 | 2/20 |
| 20260520 | 1 | 0.9395 | 10/20 | 3/20 |
| 20260521 | 1 | 0.9387 | 10/20 | 3/20 |

Per-subject, on the interrupted seed the degraded cases are inflated, not eroded
(predicted/reference volume ratio in brackets):

| Subject | Interrupted seed | Uninterrupted seed |
|---|---|---|
| IBSR_7_8 | 0.711 (1.17×) | 0.972 (1.03×) |
| IBSR_15_3 | 0.763 (1.48×) | 0.947 (0.94×) |
| IBSR_6_10 | 0.791 (1.41×) | 0.961 (1.04×) |
| IBSR_5_8 | 0.819 (1.43×) | 0.950 (0.95×) |
| IBSR_8_4 | 0.805 (1.48×) | 0.941 (0.94×) |

File timestamps confirm the mechanism: on the interrupted seed the final mask is 18–58 h
**newer** than its own `_pre-*` snapshots; on an uninterrupted seed both are written within
the same minute.

Note that interruption does **not** always corrupt a subject — it must fall at a specific
point in that subject's processing. Mindboggle-101 batch `b0` ran in 2 segments and was
unaffected (`no_base_mask 0/8`, all volume ratios 0.92–1.07).

### How to detect it

After any run, before trusting the numbers:

```python
import pandas as pd
d = pd.read_csv("<out-base>/sam_neuroadapt_<dataset>_results.csv")
suspicious = d[(d.get("inflation_guard_reason").astype(str) == "no_base_mask")
               & (d["vol_pred_cm3"] / d["vol_gs_cm3"] > 1.15)]
print(len(suspicious), "subject(s) potentially affected:", list(suspicious["subject"]))
```

A non-empty result means the inflation guard was inactive on a subject whose mask is
implausibly large. Re-run those subjects in a single uninterrupted session.

### How to avoid it

* Run each configuration **in one uninterrupted session** where possible. Published numbers
  in this repository come from uninterrupted runs except where stated.
* If a run must be resumed, **delete the output directory of the subject that was in
  progress** before resuming, so it is re-processed from a clean state.
* Always apply the detection snippet above to a resumed run before using its numbers.

---

## Issue #2 — `_pre-*` snapshots are unreliable on resumed runs (affects post-hoc analysis)

**Severity: MEDIUM — affects analysis, not the pipeline output.**

A direct consequence of Issue #1: on a resumed run, the `_pre-<stage>` files may describe an
*earlier* execution segment rather than the run that produced the final mask. Stage-by-stage
ablation tables built from those snapshots will therefore mix states from different segments
and can look internally inconsistent (e.g. every intermediate stage healthy at ~0.95 while
the final mask sits at 0.71).

We fell into this trap ourselves during analysis. Before using snapshots for ablation,
verify that their modification times are consistent with the final mask:

```bash
ls -l --time-style=full-iso <out-base>/<subject>/anat/*_desc-brain_mask*.nii.gz
```

All files for a subject should share the same processing window.

---

## Issue #3 — The adult candidate adjudicator can degrade an already-good mask

**Severity: MEDIUM. Not patched: no reliable ground-truth-free signal exists to gate it.**

`apply_adult_candidate_adjudicator()` selects the candidate that best agrees with the
registered reference atlas. When SAM2's mask is already more accurate than the registered
atlas, maximising atlas agreement pulls a good mask *towards* a slightly misregistered
atlas and degrades it.

Measured on IBSR18 (single run, 15 subjects where the adjudicator was applied): it improved
2 subjects, degraded 5, and was neutral on 8, for a mean effect of **−0.036 Dice**. The
degradations are systematic, not random — agreement with the atlas moves in the *opposite*
direction to true accuracy:

| Subject | Δ atlas agreement | Δ Dice vs. ground truth |
|---|---|---|
| IBSR_7_8 | **+0.25** | **−0.25** |
| IBSR_6_10 | +0.21 | −0.15 |
| IBSR_1_24 (helped) | −0.13 | +0.14 |

This is a concrete instance of the paper's own thesis: the atlas prior is a *failure
containment* mechanism, valuable when the foundation model fails and counter-productive
when it does not. We tested atlas agreement, mask volume and connected-component count as
gating signals; none separates the helpful from the harmful cases without ground truth.
The stage is therefore left enabled and its variance reported honestly.

---

## Issue #4 — `--no-force-extraction` across configurations

**Severity: HIGH if triggered — silent wrong results.**

`--no-force-extraction` reuses cached SAM2 masks from the output directory. Passing it on an
output directory populated by a *different* configuration silently re-scores the old masks
under the new configuration's name. During the ablation study this would have re-scored
atlas-guided masks as if they were atlas-free.

Rule: only pass `--no-force-extraction` on an output directory created by the exact same
configuration. When ablating, always use a fresh `--out-base`.

---

## Issue #5 — Surface distances in the evaluation CSVs are under-estimated

**Severity: HIGH for absolute values — reverses a between-method comparison.**

The evaluation harness computes ASSD and HD95 as the distance from boundary voxels to the
*filled volume* of the other mask, not to its *surface*
(`m1_external_adaptive_runner.py`, `dist_gs = distance_transform_edt(~gs)` evaluated at
`pred_border`). Any boundary voxel lying **inside** the other mask therefore contributes a
distance of zero. When one mask contains the other — the usual situation for an
over-segmenting method — one entire direction of the symmetric average collapses to zero.

The bias is not neutral between methods: it **rewards over-segmentation**. Measured on
Mindboggle-101 subject `Afterthought_1`, where HD-BET over-segments by 13 %:

| | harness definition | surface-to-surface |
|---|---|---|
| HD-BET ASSD | 0.681 mm | **3.237 mm** |

Recomputing all 576 masks with the standard surface-to-surface definition (real, anisotropic
voxel spacing) reverses the apparent ranking on surface metrics: with a homogeneous
definition our pipeline has *lower* ASSD and HD95 than HD-BET on all three benchmarks. The
corrected values are what the manuscript reports; they are consistent with the aggregated
error maps, which independently show over-segmentation into neck and dura for the trained
baselines.

**Detection / correction:** do not use the `assd` and `hd95` columns of the results CSVs for
between-method comparison. Recompute with `scripts/recompute_surface_metrics.py`, which
implements the standard definition and writes
`outputs/ablation_2026-07/tables/surface_metrics_corrected.csv`. Note that mixing the two
definitions in one table — recomputing for one method while reading the CSV for another —
produces a meaningless comparison; this is how the discrepancy was originally missed.

**Not patched** in the harness itself, for the reason given at the top of this document.

### A second, subtler variant: three surface definitions coexist in the analysis scripts

Even among the *post-hoc* analysis scripts, ASSD and HD95 are not computed identically:

| Script | ASSD | HD95 |
|---|---|---|
| `make_ablation_table.py` (`ablation_cascade.csv`) | mean of the **pooled** distances | 95th percentile of the **pooled** distances |
| `recompute_surface_metrics.py` (T1) / `make_table2_surface.py` (T2) | mean of the **two directional means** | **max** of the two 95th percentiles |

Both are defensible in isolation, and both use surface-to-surface distances, but they are not
interchangeable: on IBSR18 the atlas-alone configuration gives HD95 = 18.07 mm pooled versus
22.27 mm with the max-of-percentiles definition. The pooled 95th percentile is dominated by
whichever direction has more surface voxels, so it silently under-reports asymmetric errors.

`ablation_cascade.csv` therefore keeps its `assd_mean` / `hd95_mean` columns for the
stage-by-stage exploration they were written for, but **none of them appear in the
manuscript**. Every published surface distance — Tables 1 and 2, Figure 4 — comes from the
two scripts in the second row, which share one function, and each value is checked back
against the `.tex` by `scripts/verify_manuscript_numbers.py`.

**Also worth knowing:** a surface distance is undefined when a configuration produces no mask
at all. The atlas-free rows of Table 2 are therefore averaged over the subjects that did yield
a non-empty mask (BOBs 71/71, IBSR18 18/20, Mindboggle-101 27/35), which *under-states* the
degradation — the excluded cases are the worst ones. Read those cells next to the failure rate,
never on their own.

**Not patched**, same reason.

### A third variant: two "whole-head" measures with the same name

`validate_wholehead.py` reports a **ratio** — non-brain signal divided by *in-brain* signal,
unbounded above (BOBs = 1.12). `make_ablation_table.wholehead_eligibility()` reports a
**fraction** — non-brain signal divided by *total head* signal, bounded in [0, 1]
(BOBs = 0.79). Same question, different denominators, factor ~4 apart at the low end.

A draft of the manuscript quoted M-CRIB's *ratio* (0.08) inside a sentence that defined the
*fraction* (measured: 0.022 ± 0.006, n = 10). Corrected before submission; the published
sentence now quotes the fraction, and `verify_manuscript_numbers.py` checks it against the
CSV. The eligibility table also now measures M-CRIB on the 10-subject evaluation run rather
than the 1-subject diagnostic run it originally used — a dataset's eligibility should not
rest on a single image.

Neither conclusion changes: both measures place iSeg-2017 and the M-CRIB family far below
any plausible whole-head threshold, and both place the three retained benchmarks far above it.

**Fixed** in the analysis scripts (they produce no published accuracy number), and the
manuscript sentence corrected. Recorded here because the pattern — two metrics, one name —
is the same one that produced Issue #5, and it recurred despite Issue #5 being known.

---

## Issue #6 — Volume early-stop crash without atlas guidance (FIXED)

**Status: fixed** in the notebook (`sam_neuroadapt_pipeline.ipynb`,
`_propagate_mask`). Documented here because it silently invalidated an earlier ablation run.

The volume early-stop path called `slice_indices.tolist()` on a Python `range`, raising
`AttributeError: 'range' object has no attribute 'tolist'`. The branch is only reached when
the volume early-stop triggers, which never happens with atlas guidance (0/71 BOBs subjects)
but happens frequently without it — so the bug lay dormant until the atlas-free ablation, where
it caused 43/71 BOBs subjects to fail. Fixed by using `range.index()` directly, which is a
no-op for every atlas-guided run.

---

## Recommended protocol for reproducing the published numbers

1. Use the frozen seed `20260517` and the configuration flags recorded per experiment in
   `scripts/ablation_campaign.py`.
2. Run each configuration in a **single uninterrupted session**.
3. After each run, apply the Issue #1 detection snippet; discard and re-run any flagged subject.
4. For any marginal claim, run **multiple seeds** and report mean ± s.d. rather than a single
   run. On IBSR18, uninterrupted seeds agree to within ±0.0007 Dice; a single interrupted run
   fell 0.05 below that band.

## Guards present in the code

| Guard | Flag | Purpose |
|---|---|---|
| `inflation_guard` | `SAM_NEUROADAPT_INFLATION_GUARD=1` | reverts refinement that inflates volume >15 % vs. the fusion baseline. **Fails open when the baseline is missing — see Issue #1.** |
| `spatial_coherence_guard` | `SAM_NEUROADAPT_SPATIAL_GUARD=1` | reverts a stage whose output has IoU <0.40 with its own input (spatial "teleport", Dice→0.00) |
| whole-head validator | `scripts/validate_wholehead.py` | flags benchmarks distributed already skull-stripped |
| atlas-free ablation | `SAM_NEUROADAPT_NO_ATLAS=1` | disables atlas guidance at the SAM2 stage only (default off) |
