# Associated undergraduate thesis

**Title:** Advancements in Real-Time Sign Language Translation  
**Institution:** BRAC University, Department of Computer Science and Engineering  
**Date:** February 2025  
**Document type:** Undergraduate thesis, 70 PDF pages  
**Supervisor:** Farig Yousuf Sadeque

Authors, in the order printed in the thesis:

1. Eshtiak Alam Shihab
2. Md Shamsur Shafi Nur E Aziz
3. Kazi Israrul Karim
4. Tashfia Saad
5. Neelavro Shafin Qais

## Official source and citation

- [University repository record](https://dspace.bracu.ac.bd/items/1fadc573-ccaa-4928-ad29-93b2f7279b0c)
- [Persistent institutional identifier](https://hdl.handle.net/10361/26278)
- [Full metadata and rights statement](https://dspace.bracu.ac.bd/items/1fadc573-ccaa-4928-ad29-93b2f7279b0c/full)
- [BibTeX entry](references.bib): `shihab2025advancements`

Institutional deposit identifies a thesis; it does not establish peer-reviewed journal or conference publication. The `CITATION.cff` file describes the software separately from this jointly authored document.

## Relationship to the supplied implementation

The report covers literature, classification experiments, and generation-based pipelines. This repository currently contains three supplied ViViT/VideoMAE notebooks and their extracted model definitions. It does not provide source or checkpoint evidence for every experiment discussed in the thesis.

The review on **2026-11-05** identified these limits:

- The abstract's statements about superior prediction quality and real-time operation are not established by the supplied notebooks and their saved outputs.
- Chapter 7 labels cosine results as word-level similarity, while the supplied cosine implementation embeds entire sentences/sequences. Those values are not translation accuracy.
- The report discusses TimeSformer and a ResNet/CTC classification pipeline. The three supplied notebooks do not verify those implementations or their results.
- Reported latency comparisons require recoverable timing code, hardware and input specifications, preprocessing boundaries, and checkpoint lineage.
- The known sampling, token-layout, split-handling, and failed-batch problems remain documented in the [experiment review](experiment_review.md). Archived thesis tables are historical statements rather than newly validated benchmarks.

## Distribution

The university's metadata states that reproduction or distribution of its thesis reports requires written permission. This repository therefore provides the official institutional links and bibliographic metadata. The original report has not been repackaged as a newly validated paper or distributed as a repository PDF.
