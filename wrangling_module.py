"""Data wrangling and preprocessing module."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from config import Config
from exceptions import PreprocessingError

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Container for preprocessed dataset."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    dropped_columns: list[str]
    duplicates_removed: int


def remove_null_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns that are entirely null.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names).
    """
    null_cols = df.columns[df.isna().all()].tolist()
    if null_cols:
        df = df.drop(columns=null_cols)
        logger.info(f'Dropped {len(null_cols)} all-null columns: {null_cols}')
    return df, null_cols


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, count of removed rows).
    """
    count_before = len(df)
    df = df.drop_duplicates()
    removed = count_before - len(df)
    if removed > 0:
        logger.info(f'Removed {removed} duplicate rows.')
    return df, removed


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values using column median.

    Args:
        df: DataFrame (numeric columns only or mixed).

    Returns:
        DataFrame with nulls imputed for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    logger.info(f'Imputed nulls in {len(numeric_cols)} numeric columns using median.')
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns.

    Binary columns (≤2 unique): label encoding (0/1).
    Multi-category columns (>2 unique): one-hot encoding (drop_first=True).

    Args:
        df: DataFrame with categorical columns.

    Returns:
        DataFrame with all categorical columns encoded.
    """
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if not cat_cols:
        return df

    for col in cat_cols:
        unique_count = df[col].nunique()

        if unique_count <= 2:
            # Binary: label encode
            enc = LabelEncoder()
            # Fill NaN with a placeholder for encoding
            df[col] = df[col].fillna('__missing__')
            df[col] = enc.fit_transform(df[col])
            logger.info(f'Label-encoded binary column: {col}')
        else:
            # Multi-category: one-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            logger.info(f'One-hot encoded column: {col} ({unique_count} categories)')

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard scale all numeric features (zero mean, unit variance).

    Args:
        df: DataFrame with numeric features.

    Returns:
        DataFrame with scaled numeric features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f'Scaled {len(numeric_cols)} numeric features (StandardScaler).')
    return df


def preprocess(
    dataframe: pd.DataFrame,
    target_col: str,
    test_size: float = Config.TEST_SIZE,
    random_state: int = Config.RANDOM_STATE,
) -> PreprocessedData:
    """Full preprocessing pipeline.

    Steps:
    1. Remove all-null columns
    2. Remove duplicate rows
    3. Separate target from features
    4. Impute numeric nulls (median)
    5. Encode categorical columns
    6. Scale numeric features
    7. Train/test split (80/20)

    Args:
        dataframe: Input DataFrame.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed for reproducibility.

    Returns:
        PreprocessedData with train/test arrays and metadata.

    Raises:
        PreprocessingError: If zero features remain after preprocessing.
    """
    df = dataframe.copy()

    # Step 1: Remove all-null columns
    df, dropped_columns = remove_null_columns(df)

    # Step 2: Remove duplicates
    df, duplicates_removed = remove_duplicates(df)

    # Step 3: Separate target
    if target_col not in df.columns:
        raise PreprocessingError(f"Target column '{target_col}' was removed during preprocessing.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Encode target if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        enc = LabelEncoder()
        y = y.fillna('__missing__')
        y = pd.Series(enc.fit_transform(y), name=target_col)

    # Step 4: Impute numeric nulls
    X = impute_numeric(X)

    # Step 5: Encode categorical
    X = encode_categorical(X)

    # Step 6: Scale features
    X = scale_features(X)

    # Validate we have features remaining
    if X.shape[1] == 0:
        raise PreprocessingError(
            'No usable features remain after preprocessing. '
            'The dataset may lack sufficient variability.'
        )

    feature_names = X.columns.tolist()
    logger.info(f'Preprocessing complete: {X.shape[1]} features, {len(X)} rows.')

    # Step 7: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=test_size,
        random_state=random_state,
    )

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        dropped_columns=dropped_columns,
        duplicates_removed=duplicates_removed,
    )
