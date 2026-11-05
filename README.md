# Sign Language Video Modeling

Undergraduate experiments in continuous sign-language recognition and translation using pretrained video encoders, spatial attention, temporal modelling, and an autoregressive text decoder.

This repository contains reorganized thesis code by **Kazi Israrul Karim**, with readable notebooks, reusable model modules, and a record of the experiments' limitations. The historical GitHub repository name is retained; real-time performance has not been established.

## Start here

| File | Purpose |
| --- | --- |
| [Project overview](notebooks/00_project_overview.ipynb) | A lightweight walkthrough that needs no dataset, model downloads, or GPU |
| [ViViT translation](notebooks/01_vivit_translation.ipynb) | Video-to-text experiment and historical evaluation code |
| [ViViT gloss recognition](notebooks/02_vivit_gloss_recognition.ipynb) | Video-to-gloss training, sequence metrics, and embedding similarity |
| [VideoMAE experiment](notebooks/03_videomae_experiment.ipynb) | Alternative video-encoder experiment; training remains incomplete |
| [Model modules](src/sign_language_lab/) | Attention, losses, experiment-specific encoders and decoders, and path configuration |

The experiments combine a frozen pretrained video encoder with learned spatial attention, a temporal model, and a Transformer decoder. The decoder uses German BERT word embeddings; the provided code does not use BERT's full contextual encoder as its text decoder. The ViViT notebooks explore different configurations and target types, so their model variants remain separate.

```mermaid
flowchart LR
    A[Video clip] --> B[Frame preprocessing]
    B --> C[Pretrained video encoder]
    C --> D[Spatial attention]
    D --> E[Temporal model]
    E --> F[Autoregressive Transformer decoder]
    F --> G[Gloss sequence or text]
```

## Current status

The code has been reformatted and reorganized. Dataset and checkpoint paths are configurable, outputs are cleared, and the original notebook names are mapped in the [source manifest](docs/source_manifest.json).

**The historical model results have not been reproduced.** Known problems include frame sampling, ViViT token layout, evaluation split handling, and failed VideoMAE training batches. They are documented in the [experiment review](docs/experiment_review.md). Formatting and module extraction do not resolve those experimental problems. Cosine similarity is an embedding measure and must not be reported as translation accuracy.

## Local setup

Use Python 3.11 or newer for the repository tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/validate_repository.py
```

The research notebooks additionally need the packages listed in `requirements-research.in`, local PHOENIX-2014T data, and the relevant model checkpoints. That file is a dependency inventory rather than a verified training environment. See the [reproduction guide](docs/reproduction.md) before running an experiment.

## Documentation

- [Architecture and implementation](docs/architecture.md)
- [Experiment review and recorded results](docs/experiment_review.md)
- [Setup and reproduction limitations](docs/reproduction.md)
- [Changes made during organization](docs/organization_log.md)
- [Prioritized improvement plan](docs/roadmap.md)

The thesis was a group project. The implementation contribution represented by these notebooks is attributed to Kazi Israrul Karim. Dataset, pretrained-model, and library contributions belong to their respective authors. Dataset files and pretrained weights are not included.
