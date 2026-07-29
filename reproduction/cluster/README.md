# TIGRIS execution

The GitHub repository is the source of truth for code and Slurm
configuration. TIGRIS keeps a clean checkout at
`~/masters-art-repro/repo`; generated logs and results stay outside that
checkout so `git pull --ff-only` remains safe.

## One-time authentication

TIGRIS requires an RIT password and Duo Mobile. Never store either credential
in the repository. Register an SSH public key once, then verify it:

```bash
ssh-copy-id RIT_USERNAME@tigris.rc.rit.edu
ssh -o BatchMode=yes RIT_USERNAME@tigris.rc.rit.edu hostname
```

## Inspect available resources

```bash
./reproduction/cluster/probe_tigris.sh RIT_USERNAME
```

Select a TIGRIS Slurm account reported by `my-accounts` and an ARM-compatible
Spack environment containing GPU-enabled TensorFlow.

## Sync without submitting

Push the branch to GitHub first. From the repository root:

```bash
./reproduction/cluster/sync_tigris.sh \
  RIT_USERNAME \
  BRANCH
```

The first sync performs a partial-history clone of the branch. Later syncs use
`git fetch`, `git switch`, and `git pull --ff-only`. The script refuses to
overwrite a dirty TIGRIS checkout.

## Sync and submit

```bash
./reproduction/cluster/submit_tigris.sh \
  RIT_USERNAME \
  SLURM_ACCOUNT \
  SPACK_ML_ENV \
  BRANCH
```

The wrapper syncs the exact GitHub branch, submits a short GH200 smoke test,
then submits the full baseline with an `afterok` dependency. Each submission
records the Git commit and uses a unique run ID.

The default layout is:

```text
~/masters-art-repro/
├── repo/       # clean Git checkout
├── logs/       # Slurm stdout/stderr
└── runs/       # results grouped by commit and submission time
```

The current baseline allocation is one GH200 GPU, eight CPU cores, 32 GB of
CPU memory, and a two-hour wall time. Eight isolated TensorFlow workers share
the GPU; each receives one CPU thread.
