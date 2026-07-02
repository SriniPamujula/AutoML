"""File upload validation, storage, and column extraction."""

import os
import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import Config
from exceptions import ValidationError


def validate_upload(file: FileStorage) -> tuple[bool, str]:
    """Validate uploaded file is CSV and within size limit.

    Args:
        file: The uploaded file from the request.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if file is None or file.filename == '':
        return False, 'Please select a file to upload.'

    filename = file.filename or ''
    if '.' not in filename:
        return False, 'Only CSV files are accepted. Please upload a .csv file.'

    extension = filename.rsplit('.', 1)[1].lower()
    if extension not in Config.ALLOWED_EXTENSIONS:
        return False, 'Only CSV files are accepted. Please upload a .csv file.'

    return True, ''


def save_upload(file: FileStorage, upload_dir: str) -> str:
    """Save file with secure filename.

    Args:
        file: The uploaded file from the request.
        upload_dir: Directory to save the file.

    Returns:
        Full path to the saved file.

    Raises:
        ValidationError: If file cannot be saved.
    """
    if file.filename is None:
        raise ValidationError('No filename provided.')

    filename = secure_filename(file.filename)
    if not filename:
        raise ValidationError('Invalid filename.')

    # Ensure the filename stays within the upload directory
    file_path = os.path.join(upload_dir, filename)
    abs_upload_dir = os.path.abspath(upload_dir)
    abs_file_path = os.path.abspath(file_path)

    if not abs_file_path.startswith(abs_upload_dir):
        raise ValidationError('Invalid file path.')

    os.makedirs(upload_dir, exist_ok=True)
    file.save(file_path)

    return file_path


def extract_columns(file_path: str) -> list[str]:
    """Parse CSV and return column names.

    Args:
        file_path: Path to the saved CSV file.

    Returns:
        List of column names.

    Raises:
        ValidationError: If file cannot be parsed.
    """
    try:
        df = pd.read_csv(file_path, nrows=0)
        columns = list(df.columns)
        if not columns:
            raise ValidationError('CSV file has no columns.')
        return columns
    except pd.errors.EmptyDataError:
        raise ValidationError('Could not read the file. Please ensure it is a valid CSV with proper encoding.')
    except pd.errors.ParserError:
        raise ValidationError('Could not read the file. Please ensure it is a valid CSV with proper encoding.')
    except UnicodeDecodeError:
        raise ValidationError('Could not read the file. Please ensure it is a valid CSV with proper encoding.')


def detect_task_type(series: pd.Series) -> str:
    """Determine classification or regression based on target column values.

    Classification: categorical (object/bool) values OR ≤10 unique numeric values.
    Regression: continuous numeric values with >10 unique values.

    Args:
        series: The target column as a pandas Series.

    Returns:
        'classification' or 'regression'.

    Raises:
        ValidationError: If target column is all null.
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        raise ValidationError('Selected target column contains no valid data. Please choose a different column.')

    # Categorical types are always classification
    if series.dtype == 'object' or series.dtype == 'bool' or series.dtype.name == 'category':
        return 'classification'

    # Numeric with ≤10 unique values is classification
    unique_count = non_null.nunique()
    if unique_count <= 10:
        return 'classification'

    return 'regression'
