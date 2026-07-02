"""Custom exception classes for the AutoML Web Platform."""


class ValidationError(Exception):
    """Raised when input validation fails (file upload, target column, data quality)."""
    pass


class PreprocessingError(Exception):
    """Raised when data preprocessing fails (zero features, encoding errors)."""
    pass


class TrainingError(Exception):
    """Raised when model training fails (all models fail to converge)."""
    pass
