# CLIP, Bradley-Terry, and APDDv2 extensions

These experiments implement Dr. Yu's requested additions to the art paper:

1. compare CLIP ViT-B/32 with the released ResNet-50 representation;
2. compare Bradley-Terry logistic loss with the released hinge loss;
3. repeat the applicable regression and comparative simulations on APDDv2.

## Controlled protocol

- Ten seeds are used for every condition.
- Hinge and Bradley-Terry use identical splits and sampled training pairs.
- Features are standardized using training data only.
- Sidhu experiments preserve the paper's 140-example training size, add a
  20-example validation set, and use the remaining examples for testing.
- APDDv2 uses deterministic 70/15/15 train/validation/test splits.
- Pairwise objectives sweep N=1 through N=10.
- Pairwise latent scores are affine-calibrated using validation data before
  MAE and R2 are computed. Pair accuracy and rank correlations use raw scores.

APDDv2 publishes only aggregate per-image attribute scores, not individual
rater responses. Consequently, its average-rating regression and pairwise
simulations are reproducible, but the paper's within-rater and cross-rater
conditions cannot be constructed from APDDv2.

## Data-alignment correction

The released Sidhu feature arrays omit one abstract image and two
representational images and do not store item identifiers. The released
rating tables also stop before painting 240. Using array position therefore
shifts feature/rating alignment after missing images. The extension manifest
reconstructs the image IDs explicitly and joins features and ratings by ID.

## APDDv2 dependency

The official APDDv2 repository provides the annotation CSV on GitHub and the
10,023-image archive through Baidu Netdisk. The archive must be placed on
TIGRIS as:

```text
/home/xx4455/paper-projects/artifacts/comparative_painting/datasets/apddv2/
├── APDDv2-10023.csv
└── images/
    └── <archive image files>
```

The preparation job validates that every annotated filename exists before
feature extraction. The official image archive available in July 2026 omits
`36e41ae7b2764733b48475adf617b758.jpg` and includes six byte-identical
``(1)`` duplicates. The controlled manifest therefore permits and records
exactly this one missing annotation, yielding 10,022 matched images; the
unreferenced duplicates are ignored. Full images, embeddings, checkpoints,
and result CSVs remain outside Git under
`/home/xx4455/paper-projects/artifacts/comparative_painting`.

All TIGRIS jobs request one GH200 GPU and use Slack notifications for every
job state through `--mail-user=slack:@xx4455` and `--mail-type=ALL`.
