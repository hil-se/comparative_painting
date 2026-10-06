# Reproducing the October 2026 revision

Paper: *Comparative Learning for Art Aesthetics: Representation, Scale, and
Annotation Efficiency*. The controlled experiment archive comes from
`agent/clip-bt-apddv2`, commit `860c8353cff653aff63dcbd11ee124c712e3ce28`.
The screening/confirmation records and Sidhu CLIP bundle were recovered
from the original cluster archive on October 6, 2026. The held-out OLS
runner was reconstructed from the released predictors and the current
manuscript's protocol; it is not claimed to be the recovered October 1 code.

## Analysis without retraining

Install `requirements-analysis.txt` with Python 3.12, then run:

```bash
python code/extensions/run_sidhu_heldout_ols.py
python code/extensions/reproduce_art_paper.py
```

Both scripts find the repository from their own locations and accept
`--output-dir` for an alternate destination. The analysis uses the canonical
`locked_head` directories exclusively. Historical extension and original
deep-learning results elsewhere in the repository are not mixed into it.

| Paper evidence | Source | Regenerated output under `results/paper/` |
|---|---|---|
| RQ1 regression | `locked_head/aggregate_merged/{sidhu,apddv2}` | `rq1_sidhu_regression.csv`, `rq1_apdd_regression.csv` |
| Held-out OLS and sensitivity | `Data/*_Data.csv`, raw rater tables, available images | `../extensions/heldout_ols/{metrics,predictions,summary}.csv` |
| RQ1 nine-contrast Holm family | Neural regression rows plus imputed OLS | `rq1_tests.csv` |
| RQ2 N=1/N=10 focal tests | CLIP aggregate rows, macro-average within seed | `rq2_focal_tests.csv` |
| RQ2 full N=1--10 sweep | Same rows, 20 contrasts per dataset | `rq2_sweep_tests.csv`, `rq2_budget_curve.csv` |
| RQ3 aggregate/within/cross at N=1 | Sidhu CLIP aggregate plus `locked_head/rater_merged` | `rq3_n1.csv` |
| RQ3 budget curve | Rater results averaged within seed | `rq3_budget_curve.csv` |
| Head selection | `head_selection/{screen,confirm}/raw` and original sidecars | `head_{screen,confirm}_{ranking,targets}.csv` |
| Human agreement | Released `results/human_survey/survey_data` | `rq4_matrices/`, `rq4_agreement.csv` |
| Human timing | Raw Qualtrics export | `rq4_timing.csv`, `rq4_source_counts.json` |

Source paths in the first seven rows are relative to `results/extensions/`.
The figures are `art_clip_n_sweep.{png,pdf}` and
`art_rater_n_sweep.{png,pdf}`. Bands use the sample SD of ten seed-level
macro averages; conditions, targets and raters are averaged within each seed.
The older rater summary's pooled SD is not used for these bands.

The analysis verifies all 78 neural-result hashes and the entire expected
14,700-fit matrix. It rejects missing/duplicate conditions or seeds. Tests
use exact sign enumeration of signed ranks, dropping zero differences for
the test while retaining them in the mean effect size. Tied absolute
differences receive average ranks. RQ1 corrects nine contrasts jointly;
RQ2 focal tests correct three contrasts separately per dataset and budget;
RQ2 sweep tests correct 20 contrasts separately per dataset. RQ3 remains
descriptive.

## OLS provenance and leakage checks

The representational predictor CSV labels its 238 rows consecutively. The
runner restores the original painting IDs by excluding missing images 90
and 157. It independently verifies mean HSV brightness and saturation
against all 474 available predictor rows, with error below `1e-10`.
Abstract paintings 23, 121 and 154 have no predictor row; 161 and 236 lack
nonstraight edge density. Each feature median is fitted using only the
assigned 140 training paintings. Validation labels are unused. Complete-case
sensitivity filters within the same assigned partitions and never reshuffles.
Metrics, per-painting predictions, source hashes and join checks are saved.

Recomputed imputed Spearman means are 0.338599, 0.381784, 0.407664, and
0.485298 in Abstract Beauty/Liking, Representational Beauty/Liking order.
These reproduce the paper's 0.339/0.382/0.408/0.485. Complete-case abstract
means are 0.345914 and 0.387246. All nine RQ1 contrasts give Holm
`p=0.017578125`.

## Prepare inputs for neural training

The analysis environment is a tested CPU environment, not a recovered lock
of the original cluster. Install `requirements-training.txt` for the neural
runners and feature extraction. TensorFlow 2.16.1 is recorded in the fit
sidecars. The Torch/Transformers pins are an installation recipe; their
original versions were not recorded. Hardware/library changes can affect
new neural fits. Archived result regeneration is independent of those fits.

Prepare Sidhu from the repository:

```bash
python code/extensions/build_art_manifests.py --dataset sidhu \
  --output build/manifests/sidhu.csv \
  --resnet-output build/features/sidhu-resnet50.npz
```

The builder returns 477 images with original painting IDs. Use the recovered
`Data/fixed_features/sidhu-clip-vit-b32.npz` for CLIP. Its SHA-256 is
`2482b5c2ba3cc4c7c8cf51d8384300b6afa118a0e2e9dc667b48ea4049de4ce8`.
To extract fresh CLIP features instead, use `extract_art_features.py`; it
pins `openai/clip-vit-base-patch32` to revision
`8092f5b35a22023f7a822152e20837ac59cb91a3`.

Obtain APDDv2's annotations and image archive from the
[official dataset repository](https://github.com/BestiVictory/APDDv2).
The authors specify CC BY-NC-ND 4.0. Dataset files are obtained from that
source rather than redistributed here. Prepare them with:

```bash
python code/extensions/build_art_manifests.py --dataset apddv2 \
  --annotations /path/to/APDDv2-10023.csv --images /path/to/images \
  --max-missing-images 1 --output build/manifests/apddv2.csv
python code/extensions/extract_art_features.py \
  --manifest build/manifests/apddv2.csv --representation clip-vit-b32 \
  --output build/features/apddv2-clip-vit-b32.npz --batch-size 64
python code/extensions/extract_art_features.py \
  --manifest build/manifests/apddv2.csv --representation resnet50 \
  --output build/features/apddv2-resnet50.npz --batch-size 64
```

The expected image count is 10,022; per-target valid counts range from
3,818 to 10,022. Only finite, nonempty target labels enter each split.
Original APDDv2 feature hashes are in `Data/provenance/`. Historical
manifest hashes include absolute cluster paths and will differ when a
manifest is rebuilt locally. Item IDs, labels and feature alignment remain
the matching criteria; do not overwrite original provenance hashes.

## Run the controlled models

The README gives a complete aggregate Sidhu command. Repeat it for both
painting categories, both targets, and both representations. Use the
released legacy ResNet features for Sidhu; freshly extracting standardized
ResNet features would change the reported protocol. For APDDv2, use
`--dataset apddv2 --category all` and each of the 11 target names listed in
`build_art_manifests.APDD_TARGETS`, with both representations. These runners
use seeds 0--9 and N=1--10 by default.

Rater-level example:

```bash
python code/extensions/run_sidhu_rater_locked_head.py \
  --features Data/fixed_features/sidhu-clip-vit-b32.npz --data-dir Data \
  --category abstract --target beauty --mode within --rater 1 \
  --objectives regression,hinge,bradley_terry --n-values 1-10 --seeds 0-9 \
  --output build/reruns/sidhu-abstract-beauty-within-r1.csv
```

Repeat for `within` and `cross`, raters 1--5, both categories and both
targets. The shared head remains locked; a new run is not used to retune it.
Cluster jobs in `jobs/tigris/` retain their historical account and paths;
the Python commands above do not depend on those cluster settings.

The original pair sampler shuffles anchors and eligible partners, rejecting
ties and reused unordered pairs. It is not a uniform draw over all possible
pairs. Every archived pairwise fit reached `N * train_examples`; the
implementation allows shortfalls when eligible partners are exhausted.
The repository documents this behavior and preserves the experimental code.

## Human-study accounting

The raw file has seven finished entries, one marked Survey Preview. After
excluding the preview there are six completed responses. The timing filter
excludes constant beauty-rating responder P2 and retains five; the agreement
panel uses P1/P3/P4/P5/P6. The manuscript's statement that two completed
participants were removed for insufficient variance is not demonstrated by
this export. The source counts are recorded rather than inferred.

Human-to-human agreement averages the ten unique retained-rater pairs and
excludes GT. Human-to-GT agreement averages five pairs separately. The
matrices round each pair's accuracy to two decimals before summary averaging,
matching the manuscript's summaries (0.725 direct, 0.520 comparative).
The analysis recognizes both generic and target-specific GT score headers;
the original script skipped the latter for Representational Beauty.

Timing retains full precision in `seconds_raw`. The paper's displayed overall
values average the condition means after rounding them to two decimals;
`seconds` reproduces that convention (27.28/10.71). The unrounded direct
grand mean is 27.27042 seconds. Timings describe the retained sample and do
not establish a population-level effect or remove task-order confounds.
