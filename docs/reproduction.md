# Setup and reproduction

## Read the project without research dependencies

The overview notebook uses only Python's standard library and the lightweight path configuration module. It can display the notebook index and recorded-results summary without importing PyTorch or downloading weights.

To validate the repository, install `requirements-dev.txt` and run:

```bash
python scripts/validate_repository.py
python -m black --check src scripts
python -m ruff check src scripts
python -m compileall -q src
```

These checks validate organization and syntax, not model training.

## Research environment

The original package versions were not recorded in the supplied notebooks. `requirements-research.in` lists observed dependencies; compatibility must be established before producing a lock file. In particular, the model-loading and MediaPipe APIs may depend on historical versions. The research notebooks use CUDA-oriented calls and expect local data.

The original experiments used customized Excel annotations and converted MP4 video files. The workbook, conversion pipeline, trained checkpoints, and a verified filename-to-annotation manifest were not supplied with these notebooks. Downloading the public dataset alone does not recreate those local artifacts.

## Configure local data

Before starting Jupyter, export:

```bash
export PHOENIX_DATA_ROOT="/path/to/phoenix2014t"
export PHOENIX_ANNOTATIONS_FILE="/path/to/annotations.xlsx"
```

The expected local video layout is:

```text
phoenix2014t/
  videos/
    train/
    dev/
    test/
  annotations.xlsx
```

The notebooks retain their original workbook sheet names and target-column assumptions. Check them against your workbook rather than renaming labels to conceal a target mismatch. A secondary historical development-CSV variable maps to `annotations/dev.csv` under the dataset root.

Outputs now live under separate directories:

```text
artifacts/
  vivit_translation/
  vivit_gloss_recognition/
  videomae_experiment/
```

Several original training cells load an existing checkpoint before training. They will still require that checkpoint. Do not copy an unidentified historical `best_model.pth` into every experiment directory; establish its model configuration and origin first.

## Before reporting a new result

1. Fix the known data, sampling, token-layout, and failed-batch problems.
2. Create explicit train/dev/test manifests and check overlap, including duplicate content.
3. Run from a fresh kernel using a recorded environment and fixed seed.
4. Verify that a small training run processes actual batches and can overfit a tiny sample.
5. Select checkpoints on development data and evaluate a frozen test split.
6. Save predictions, references, split identity, checkpoint hash, code commit, metric settings, and sample counts together.
7. Report unfavorable task metrics alongside any supplementary embedding similarity.

No full run has been carried out as part of this code-organization update.
