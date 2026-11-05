# Architecture

The three experiments share attention and loss definitions but retain distinct encoder and decoder configurations. Refactoring keeps those differences visible rather than silently choosing a single variant.

| Module | Contents |
| --- | --- |
| `attention.py` | Upper-body attention, general spatial attention, positional encodings, temporal attention |
| `losses.py` | Cross-entropy wrapper, label smoothing, consistency loss |
| `vivit_translation.py` | Translation notebook's skip connection, ViViT encoder, decoder, and wrapper |
| `vivit_gloss_recognition.py` | Gloss notebook's skip connection, ViViT encoder, decoder, and wrapper |
| `videomae_experiment.py` | VideoMAE notebook's skip connection, encoder, decoder, and wrapper |
| `config.py` | Environment-based dataset, annotation, and output paths |

## Video encoders

The ViViT experiments use `google/vivit-b-16x2-kinetics400`. Their intended design aggregates spatial patch information and models the resulting temporal sequence with a BiLSTM or another selected temporal branch. Preprocessing supplies fixed-length clips. Existing sampling and token-reshape problems remain documented in the experiment review.

The VideoMAE experiment uses `MCG-NJU/videomae-base`. It contains an attempted analogous temporal pipeline, but its saved final training run does not demonstrate successful learning.

## Decoder

The code loads a German BERT model and tokenizer, then uses the model's word-embedding table. Text decoding is performed by the separately defined PyTorch `TransformerDecoder`, with causal masking and next-token training. Beam search generates candidate sequences.

This implementation does not establish an end-to-end real-time system, a new state-of-the-art architecture, or validated translation performance. No TimeSformer, Swin, or CTC implementation is verified by these three source notebooks.

## Why training remains in notebooks

Training and evaluation helpers still depend on experiment-local variables and historical checkpoint assumptions. They remain beside their corresponding configuration cells for review. Extracting those loops into a single executable training application requires an additional behavior-changing refactor and fresh validation.
