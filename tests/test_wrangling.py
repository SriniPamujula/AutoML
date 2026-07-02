"""Tests for wrangling_module."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wrangling_module import (
    remove_null_columns, remove_duplicates, impute_numeric,
    encode_categorical, scale_features, preprocess
)
from exceptions import PreprocessingError


class TestRemoveNullColumns:
    def test_removes_all_null_columns(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [None, None], 'c': [3, 4]})
        result, dropped = remove_null_columns(df)
        assert 'b' in dropped
        assert 'b' not in result.columns
        assert 'a' in result.columns
        assert 'c' in result.columns

    def test_no_null_columns(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result, dropped = remove_null_columns(df)
        assert dropped == []
        assert list(result.columns) == ['a', 'b']


class TestRemoveDuplicates:
    def test_removes_duplicates(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [10, 10, 20]})
        result, count = remove_duplicates(df)
        assert count == 1
        assert len(result) == 2

    def test_no_duplicates(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        result, count = remove_duplicates(df)
        assert count == 0
        assert len(result) == 3


class TestImputeNumeric:
    def test_imputes_with_median(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = impute_numeric(df)
        assert result['a'].isna().sum() == 0
        assert result['a'].iloc[1] == 3.0  # median of [1,3,5]

    def test_no_nulls_unchanged(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        result = impute_numeric(df)
        assert list(result['a']) == [1.0, 2.0, 3.0]


class TestEncodeCategorical:
    def test_binary_label_encoded(self):
        df = pd.DataFrame({'gender': ['M', 'F', 'M', 'F']})
        result = encode_categorical(df)
        assert set(result['gender'].unique()).issubset({0, 1})

    def test_multi_category_one_hot(self):
        df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red']})
        result = encode_categorical(df)
        assert 'color' not in result.columns
        assert result.shape[1] == 2  # 3 cats - 1 (drop_first)


class TestScaleFeatures:
    def test_zero_mean(self):
        df = pd.DataFrame({'a': [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = scale_features(df)
        assert abs(result['a'].mean()) < 1e-10


class TestPreprocess:
    def test_full_pipeline(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
            'target': ['A', 'B'] * 25,
        })
        result = preprocess(df, 'target')
        assert result.X_train.shape[0] == 40
        assert result.X_test.shape[0] == 10
        assert result.X_train.shape[1] > 0

    def test_raises_on_zero_features(self):
        df = pd.DataFrame({'null_col': [None] * 20, 'target': range(20)})
        with pytest.raises(PreprocessingError):
            preprocess(df, 'target')
