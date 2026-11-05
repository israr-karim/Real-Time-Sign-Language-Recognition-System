# Experiment review

Review date: **2026-11-05**. The review used the three original notebook files, their saved outputs, and isolated helper checks. It did not reproduce training or obtain the external `best_model.pth` checkpoints. Cell numbers below refer to the original files; recreated notebook cells retain `original_cell_number` metadata.

## Recorded results

These are historical observations, not newly verified benchmarks. BLEU is shown on the original 0–1 scale; WER is an error rate, not accuracy. Embedding cosine is reported separately and is not an accuracy percentage.

| Original experiment | Saved evaluation | WER | BLEU | Mean cosine | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `Vivit2.0_fff.ipynb` | 500 examples loaded through the training-data path | 0.1171 | 0.7481 | — | Unreliable as held-out evidence; split and constructor bugs |
| ViViT gloss notebook | 519 development examples | 0.9485 | 0.0092 | ~0.6159 | Poor recorded gloss recognition; not reproduced |
| ViViT gloss notebook | 500 training examples | 0.9384 | 0.0110 | ~0.6007 | Poor recorded gloss recognition; not reproduced |
| ViViT gloss notebook | 642 test examples | 0.9472 | 0.0163 | ~0.6073 | Poor recorded gloss recognition; not reproduced |
| `VideoMAE.ipynb` | Final 100-epoch training attempt | — | — | — | 200 training-batch errors; zero loss is not successful learning |

The gloss notebook trains on the `gloss` column. Gloss sequences annotate signs; they are a different target from natural-language sentence translation. Its cosine code embeds whole sentences/sequences rather than computing word-by-word alignment.

All available task metrics should be included when discussing performance. Choosing embedding similarity because sequence metrics are unfavorable does not establish recognition or translation quality.

## Findings affecting validity

### 1. Sampling truncates long inputs

Both ViViT samplers select 32 indices from 167 to 199 for a 200-frame video when called with `clip_len=32` and `frame_sample_rate=1`. Much of the video is therefore absent while the full sequence label remains. The provided fixed-length preprocessing also prevents the encoder's longer-input sliding-window branch from being exercised.

### 2. ViViT token geometry is misinterpreted

For 32 input frames, the checkpoint's two-frame tubelets and 16-by-16 spatial patches produce 16 temporal groups of 196 patch tokens, plus one classification token. The encoder instead groups 3136 retained tokens into 32 groups of 98, retaining the classification token and discarding a patch. This breaks the intended interpretation of spatial attention. See the [checkpoint configuration](https://huggingface.co/google/vivit-b-16x2-kinetics400/blob/main/config.json).

### 3. Translation notebook evaluation uses the wrong data

Original cell 29 uses the first ten training examples and the first two of those same examples for validation. That can serve an overfitting/debugging exercise but cannot measure generalization. Later cells load training paths and training-sheet labels for evaluation. In original cell 39, the dataset constructor assigns global `video_paths` and `sentences` instead of its supplied arguments. An isolated check confirmed that passed examples are ignored.

Execution counts and checkpoint reuse prevent reconstructing a single trustworthy run history. The high saved score cannot be attributed confidently to a particular current training cell. The provided encoder/decoder dimensions and the validation projection path also need reconciliation.

### 4. VideoMAE errors are counted as zero-loss epochs

The processing function expects one video path and an image processor; the training call supplies a batch and an unsupported keyword. The saved run records 200 training errors and 200 validation error lines across 100 epochs. Failed batches are skipped, leaving a zero-valued loss accumulator. Validation also references a video-processing helper not defined in that source notebook.

### 5. Training bookkeeping needs a separate repair

The gloss notebook clips gradients after an optimizer step and advances its learning-rate schedule at a different cadence from accumulated optimizer updates. Checkpoint selection and failed-batch handling also need explicit policies. These findings have not been repaired merely by reorganizing source files.

## What has been verified during organization

- Notebook schema validity and Python syntax.
- Consistent code formatting.
- Extracted class bodies against their original syntax trees.
- Configurable paths and distinct output directories.
- Removal of stored notebook outputs and execution counters.

No new model-quality, latency, or reproducibility claim follows from those checks.
