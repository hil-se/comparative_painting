# Art-paper baseline reproduction

## Outcome

The repository baseline is operational and reproduced closely enough to use as
the reference implementation for the new experiments.

- Repository: `hil-se/comparative_painting`
- Commit: `b151e2e28870d836d7adaa120e9fc684acd94018`
- Original model functions: unchanged
- Features: repository's resized ResNet50 features (`origin=False`)
- Split: 140 training images, remaining images for testing
- Repetitions: fixed seeds 0–9
- TIGRIS smoke job: `22260` (`COMPLETED`, exit `0:0`)
- TIGRIS full job: `22262` (`COMPLETED`, exit `0:0`, elapsed 7m23s)
- Hardware: one NVIDIA GH200, eight CPU cores
- Software: RIT Spack environment `default-ml-aarch64-25050701`,
  TensorFlow 2.16.1
- Validation: 120/120 successful rows, 120 unique experiment keys, no
  failures, no missing rows, no duplicates

The deterministic OLS script also reproduced the repository's tracked values
to floating-point precision.

## Direct regression

Each cell is `reproduced / repository`. Values are means across ten runs.

| Painting | Rating | MAE | R² | Pearson ρ | Spearman rₛ |
|---|---|---:|---:|---:|---:|
| Abstract | Beauty | 0.6511 / 0.6161 | 0.3212 / 0.3847 | 0.5922 / 0.6490 | 0.5459 / 0.5938 |
| Abstract | Liking | 0.5306 / 0.5068 | 0.2037 / 0.2770 | 0.4882 / 0.5631 | 0.4241 / 0.5020 |
| Representational | Beauty | 0.5818 / 0.5604 | 0.3279 / 0.3435 | 0.6095 / 0.6310 | 0.6068 / 0.6173 |
| Representational | Liking | 0.6101 / 0.5875 | 0.3646 / 0.4286 | 0.6229 / 0.6664 | 0.6097 / 0.6581 |

## Comparative hinge model

Each cell is `reproduced / repository`. MAE and R² are omitted because the
encoder produces latent utilities that are not calibrated to the human rating
scale.

| Painting | Rating | N=1 Pearson ρ | N=1 Spearman rₛ | N=10 Pearson ρ | N=10 Spearman rₛ |
|---|---|---:|---:|---:|---:|
| Abstract | Beauty | 0.5402 / 0.5516 | 0.5154 / 0.5195 | 0.5729 / 0.6167 | 0.5179 / 0.5692 |
| Abstract | Liking | 0.4538 / 0.4939 | 0.4146 / 0.4447 | 0.5111 / 0.5390 | 0.4389 / 0.4703 |
| Representational | Beauty | 0.5531 / 0.5276 | 0.5469 / 0.5233 | 0.6020 / 0.6727 | 0.6118 / 0.6790 |
| Representational | Liking | 0.5429 / 0.4877 | 0.5511 / 0.4772 | 0.6470 / 0.6654 | 0.6395 / 0.6568 |

## Interpretation

- The mean absolute difference in Pearson/Spearman correlation across the
  reproduced and repository means is 0.0406; the maximum is 0.0779.
- For the primary metrics—every regression metric plus comparative
  correlations—31 of 32 repository differences are no larger than one
  standard deviation of the reproduced runs. This is a descriptive
  run-to-run comparison, not a formal significance test.
- Exact neural equality is not expected because the repository does not record
  the authors' random seeds and TIGRIS uses a newer TensorFlow/CUDA stack.
- Comparative MAE and R² are extremely negative because pairwise hinge
  utilities have an arbitrary offset and are not calibrated to the 1–9 rating
  scale. Future hinge-versus-Bradley–Terry comparisons should use Pearson,
  Spearman, pairwise accuracy, and/or fit a calibration mapping on training
  data before reporting MAE/R².
- Average-rating reproduction is unaffected by rater selection. Before
  within-rater and cross-rater experiments, the intended cohort must be
  resolved: the paper describes five raters, the regression runner selects ten,
  and the bundled CSV files contain 43–49 individual rater columns.

## Next implementation step

Create one persisted split manifest for seeds 0–9 and make every condition use
the same train/test indices. Then extract CLIP embeddings on TIGRIS and run a
paired matrix:

1. ResNet50 versus CLIP features;
2. direct regression, hinge comparison, and Bradley–Terry comparison;
3. abstract/representational × beauty/liking;
4. the same metrics and identical splits for every method.

This paired design isolates the effect of representation and loss function
instead of confounding it with random data splits.
