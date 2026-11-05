# Organization log

## 2026-11-05

- Recreated three source notebooks with descriptive filenames, section labels, consistent indentation, and cleared outputs.
- Extracted identical attention and loss classes into shared modules.
- Kept differing ViViT and VideoMAE model variants in separate modules, preserving their class bodies.
- Replaced inline class definitions in notebooks with explicit imports.
- Added environment-based data paths and separate experiment output directories, replacing workstation-specific paths and a shared relative checkpoint filename.
- Converted ad hoc notebook `pip install` cells into setup notes; dependencies are documented centrally.
- Omitted empty and comment-only scratch cells, including a large commented list of reference sentences.
- Split long cells at top-level function/class boundaries while retaining execution order.
- Added an original-to-recreated filename map, source hashes, and definition provenance.
- Added a standard-library overview notebook and formatting/schema validation tooling.
- Documented historical metrics, known limitations, and the next experimental repairs.
- Made definition fingerprints independent of Python's `ast.dump` display format, and added hosted validation on Python 3.11 and 3.14.

The original downloaded notebook files were left unchanged. Original output streams are not included; aggregate observations are preserved in the experiment review and results JSON. No training result was regenerated or improved by this organization work.
