# Dataset and local preparation

## Upstream source

- **Dataset:** RWTH-PHOENIX-Weather 2014T.
- **Official download and citation page:** [RWTH Aachen](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).
- **Dataset paper:** [Neural Sign Language Translation, Camgoz et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html).
- **Scope:** German Sign Language weather broadcasts, with gloss annotations and German text translations.

Use the **2014T** release for these experiments. The dataset paper explains that its segmentation differs from PHOENIX-2014, so versions and splits must be identified explicitly. Bibliographic details are included in [references.bib](references.bib).

## Local preparation expected by this repository

The historical experiments use converted MP4 files and customized Excel workbooks. Those artifacts and their preparation scripts were not supplied with the notebooks. The current path configuration expects `videos/train`, `videos/dev`, and `videos/test` beneath `PHOENIX_DATA_ROOT`, plus an explicit `PHOENIX_ANNOTATIONS_FILE`.

To establish a reproducible preparation process, record:

1. Upstream release/archive identity and SHA-256 checksum.
2. Conversion script revision, frame order, frame rate, resizing, and any pose overlays.
3. A manifest mapping every local video to its upstream sample ID and official split.
4. Explicit target type: gloss sequence or German translation.
5. Missing, excluded, duplicate, or renamed samples and the reason for each change.

Do not assume the locally selected examples match the official benchmark splits. That must be verified from the preparation manifest.

## Usage and redistribution status

Checked on **2026-11-05**: the official landing page provides access and a citation, but the review did not establish explicit redistribution terms there. Its linked standalone README files returned HTTP 404. Inspect the terms supplied with the actual release or obtain clarification from the maintainers before redistributing dataset material.

This repository links to the upstream source. Dataset files, customized annotations, and model artifacts remain excluded from Git. No dataset license is invented or inferred from its availability for download.
