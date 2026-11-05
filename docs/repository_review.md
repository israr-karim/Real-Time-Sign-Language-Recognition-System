# Repository-practice review

Reviewed on **2026-11-05** against the linked publisher guidance, [GitHub citation guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files), and [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

| Area | Current evidence | Status |
| --- | --- | --- |
| File organization | Named notebooks, shared modules, project overview | Established |
| Formatting and notebook hygiene | Black, Ruff, schema checks, cleared outputs | Automated validation |
| Original-code provenance | Source hashes, cell mapping, extracted-class fingerprints | Recorded and checked |
| Dataset identification | Official source, target type, version distinction, preparation gaps | Documented |
| Attribution | Dataset/backbone bibliography and checkpoint source inventory | Added |
| Software citation | `CITATION.cff`, without an invented DOI or publication claim | Added |
| Original-code licensing | No license selected | Owner decision remains |
| Dataset/model redistribution | No bundled data or weights; checkpoint terms listed | Dataset terms still need verification |
| Project manuscripts | Confirmed as unpublished university-submitted drafts; titles, author lists, files, and versions still pending | Status established; files await review |
| Training reproducibility | Missing original environment, conversion artifacts, run/checkpoint lineage | Incomplete |
| Scientific validity | Known sampling, token-layout, split, and failed-batch defects | Unresolved |

The repository supports review of implementation work. A reproducible research release still needs the substantive repairs and fresh experiments described in [the improvement plan](roadmap.md). Passing code-format and schema checks does not establish model quality or complete compliance with every publisher's conditions.
