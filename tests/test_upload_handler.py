"""Tests for upload_handler module."""

import os
import tempfile
import pandas as pd
import pytest
from io import BytesIO
from werkzeug.datastructures import FileStorage

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from upload_handler import validate_upload, save_upload, extract_columns, detect_task_type
from exceptions import ValidationError


class TestValidateUpload:
    def test_rejects_none_file(self):
        valid, msg = validate_upload(None)
        assert not valid
        assert 'select a file' in msg.lower()

    def test_rejects_empty_filename(self):
        file = FileStorage(stream=BytesIO(b'data'), filename='')
        valid, msg = validate_upload(file)
        assert not valid

    def test_rejects_non_csv(self):
        file = FileStorage(stream=BytesIO(b'data'), filename='test.xlsx')
        valid, msg = validate_upload(file)
        assert not valid
        assert 'CSV' in msg

    def test_accepts_csv(self):
        file = FileStorage(stream=BytesIO(b'a,b\n1,2'), filename='test.csv')
        valid, msg = validate_upload(file)
        assert valid
        assert msg == ''

    def test_rejects_no_extension(self):
        file = FileStorage(stream=BytesIO(b'data'), filename='noext')
        valid, msg = validate_upload(file)
        assert not valid


class TestSaveUpload:
    def test_saves_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = b'col1,col2\n1,2\n3,4'
            file = FileStorage(stream=BytesIO(content), filename='test.csv')
            path = save_upload(file, tmpdir)
            assert os.path.exists(path)
            assert path.endswith('test.csv')

    def test_secure_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = FileStorage(stream=BytesIO(b'data'), filename='../../etc/passwd.csv')
            path = save_upload(file, tmpdir)
            assert 'etc' not in path
            assert tmpdir in os.path.abspath(path)


class TestExtractColumns:
    def test_extracts_columns(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('col_a,col_b,col_c\n1,2,3\n')
            f.flush()
            cols = extract_columns(f.name)
            assert cols == ['col_a', 'col_b', 'col_c']
            os.unlink(f.name)

    def test_rejects_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')
            f.flush()
            with pytest.raises(ValidationError):
                extract_columns(f.name)
            os.unlink(f.name)


class TestDetectTaskType:
    def test_categorical_is_classification(self):
        s = pd.Series(['A', 'B', 'C', 'A'])
        assert detect_task_type(s) == 'classification'

    def test_few_unique_numeric_is_classification(self):
        s = pd.Series([0, 1, 0, 1, 0, 1])
        assert detect_task_type(s) == 'classification'

    def test_ten_unique_is_classification(self):
        s = pd.Series(list(range(10)) * 5)
        assert detect_task_type(s) == 'classification'

    def test_many_unique_numeric_is_regression(self):
        s = pd.Series(range(100))
        assert detect_task_type(s) == 'regression'

    def test_all_null_raises(self):
        s = pd.Series([None, None, None])
        with pytest.raises(ValidationError):
            detect_task_type(s)

    def test_boolean_is_classification(self):
        s = pd.Series([True, False, True, False])
        assert detect_task_type(s) == 'classification'
