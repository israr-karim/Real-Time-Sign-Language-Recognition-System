# Third-party resources

Resource inventory checked on **2026-11-05**. Checkpoint license labels below are those reported by the linked model cards; consult their actual terms for a particular use. Downloaded weights and dataset files are not bundled with this repository.

| Resource | Role in the supplied code | Reported checkpoint license |
| --- | --- | --- |
| [google/vivit-b-16x2-kinetics400](https://huggingface.co/google/vivit-b-16x2-kinetics400) | Pretrained video backbone | MIT |
| [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) | Alternative pretrained video backbone | CC-BY-NC-4.0 |
| [google-bert/bert-base-german-cased](https://huggingface.co/google-bert/bert-base-german-cased) | German tokenizer and word embeddings; original code uses the short model ID | MIT |
| [sentence-transformers/distilbert-base-nli-stsb-mean-tokens](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens) | Supplementary sentence-embedding cosine calculation | Apache-2.0; model card marks the checkpoint deprecated |
| [RWTH-PHOENIX-Weather 2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | Source dataset | Redistribution terms not established in this review; see [dataset guide](docs/dataset.md) |

The VideoMAE checkpoint's noncommercial terms must be assessed independently of any eventual license selected for the project's original code. Paper, model-weight, library-code, and dataset licenses can differ even when the resources are used in one pipeline.

Research-package imports are listed in `requirements-research.in`; their transitive licenses have not been audited. Original dependency versions and checkpoint revision hashes also remain to be recovered or fixed for a future reproducible run.

The accompanying [bibliography](docs/references.bib) records the dataset and video-backbone papers. The German BERT and sentence-transformer cards provide further authorship and citation information.
