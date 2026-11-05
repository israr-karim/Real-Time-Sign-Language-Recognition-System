"""Portable paths for the three historical experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolve local data and separate each experiment's generated artifacts."""

    dataset_root: Path
    annotations_file: Path
    output_dir: Path

    @classmethod
    def from_environment(cls, experiment_name: str) -> ExperimentPaths:
        """Read explicit shell variables, falling back to repository-local paths."""
        if not experiment_name or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in experiment_name
        ):
            raise ValueError("Use a lowercase experiment name with underscores.")

        repository_root = Path(__file__).resolve().parents[2]
        dataset_root = (
            Path(
                os.environ.get(
                    "PHOENIX_DATA_ROOT", repository_root / "data" / "phoenix2014t"
                )
            )
            .expanduser()
            .resolve()
        )
        annotations_file = (
            Path(
                os.environ.get(
                    "PHOENIX_ANNOTATIONS_FILE", dataset_root / "annotations.xlsx"
                )
            )
            .expanduser()
            .resolve()
        )
        output_root = (
            Path(
                os.environ.get(
                    "SIGN_LANGUAGE_OUTPUT_ROOT", repository_root / "artifacts"
                )
            )
            .expanduser()
            .resolve()
        )
        return cls(dataset_root, annotations_file, output_root / experiment_name)

    @property
    def train_dir(self) -> Path:
        return self.dataset_root / "videos" / "train"

    @property
    def dev_dir(self) -> Path:
        return self.dataset_root / "videos" / "dev"

    @property
    def test_dir(self) -> Path:
        return self.dataset_root / "videos" / "test"

    @property
    def checkpoint_file(self) -> Path:
        return self.output_dir / "best_model.pth"
