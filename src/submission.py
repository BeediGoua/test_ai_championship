"""Helpers for building and writing ordered prediction submissions."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_submission(
    identifiers: Any,
    predictions: Any,
    *,
    id_column: str = "id",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Build a two-column submission, preserving the supplied identifier order."""
    ids = np.asarray(identifiers)
    values = np.asarray(predictions)

    if ids.ndim != 1 or values.ndim != 1:
        raise ValueError("identifiers and predictions must be one-dimensional.")
    if ids.size == 0 or ids.size != values.size:
        raise ValueError(
            "identifiers and predictions must have the same non-zero length."
        )
    if pd.isna(ids).any() or pd.Index(ids).has_duplicates:
        raise ValueError("identifiers must be non-null and unique.")
    try:
        numeric_predictions = values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("predictions must be numeric.") from error
    if not np.isfinite(numeric_predictions).all():
        raise ValueError("predictions must contain only finite values.")

    return pd.DataFrame(
        {id_column: ids, prediction_column: numeric_predictions}
    )


def write_submission(
    submission: pd.DataFrame,
    path: str | PathLike[str] = "outputs/predictions.csv",
    *,
    id_column: str = "id",
    prediction_column: str = "prediction",
) -> Path:
    """Write a validated submission CSV and return its path."""
    required_columns = [id_column, prediction_column]
    if list(submission.columns) != required_columns:
        raise ValueError(
            f"submission must have exactly the columns {required_columns!r}."
        )
    if submission.empty:
        raise ValueError("submission must contain at least one prediction.")
    if (
        submission[id_column].isna().any()
        or submission[id_column].duplicated().any()
    ):
        raise ValueError("submission identifiers must be non-null and unique.")
    try:
        predictions = submission[prediction_column].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("submission predictions must be numeric.") from error
    if not np.isfinite(predictions).all():
        raise ValueError("submission predictions must be finite.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path
