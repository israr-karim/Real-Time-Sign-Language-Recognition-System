# Improvement plan

## Completed in this organization pass

- [x] Use clear notebook and module names.
- [x] Format Python consistently and separate shared model components.
- [x] Remove workstation-specific data paths and stored notebook outputs.
- [x] Document targets, recorded results, and known limitations.
- [x] Record the relationship between original and recreated files.

## Required before a credible benchmark

- [ ] Establish the dataset conversion and annotation mapping.
- [ ] Validate disjoint train/dev/test manifests and duplicate-content checks.
- [ ] Replace end-of-clip-only sampling with an explicit, tested temporal policy.
- [ ] Correct ViViT tubelet layout and classification-token handling.
- [ ] Reconcile encoder/decoder dimensions and training/inference projections.
- [ ] Repair VideoMAE preprocessing signatures and missing validation helpers.
- [ ] Fail clearly when no batch succeeds; record processed/skipped counts.
- [ ] Correct gradient clipping, optimizer/scheduler cadence, and checkpoint selection.
- [ ] Freeze and record a compatible environment, seed, and model revision.
- [ ] Re-run one simple baseline before expanding model complexity.
- [ ] Publish all task metrics with predictions, references, and run provenance.

## Later development

- [ ] Move training/evaluation loops into explicit, configurable entry points.
- [ ] Compare a small number of controlled architectural variants.
- [ ] Measure end-to-end latency before making a real-time claim.
