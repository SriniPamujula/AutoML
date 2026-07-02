"""Exploratory Data Analysis module."""

import logging
from dataclasses import dataclass

import pandas as pd

from config import Config
from exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class EDAResults:
    """Container for EDA analysis results."""

    head: pd.DataFrame          # First 5 rows
    shape: tuple[int, int]      # (rows, columns)
    describe: pd.DataFrame      # Descriptive statistics
    null_counts: pd.Series      # Null count per column
    dtypes: pd.Series           # Data type per column
    unique_counts: pd.Series    # Unique values per column


def analyze(dataframe: pd.DataFrame, target_col: str) -> EDAResults:
    """Compute full EDA summary for the dataset.

    Args:
        dataframe: The input DataFrame.
        target_col: The selected target column name.

    Returns:
        EDAResults with all computed metrics.

    Raises:
        ValidationError: If target column doesn't exist, is all null, or dataset too small.
    """
    # Validate target column exists
    if target_col not in dataframe.columns:
        raise ValidationError(f"Target column '{target_col}' not found in dataset.")

    # Validate target column is not all null
    if dataframe[target_col].isna().all():
        raise ValidationError(
            'Selected target column contains no valid data. Please choose a different column.'
        )

    # Validate minimum row count
    if len(dataframe) < Config.MIN_ROWS:
        raise ValidationError(
            f'Dataset has fewer than {Config.MIN_ROWS} rows. '
            'Please provide more data for meaningful analysis.'
        )

    logger.info(f'Running EDA on dataset with shape {dataframe.shape}, target: {target_col}')

    # Compute EDA metrics
    head = dataframe.head(min(5, len(dataframe)))
    shape = dataframe.shape
    describe = dataframe.describe(include='all')
    null_counts = dataframe.isna().sum()
    dtypes = dataframe.dtypes
    unique_counts = dataframe.nunique()

    return EDAResults(
        head=head,
        shape=shape,
        describe=describe,
        null_counts=null_counts,
        dtypes=dtypes,
        unique_counts=unique_counts,
    )
