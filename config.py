"""Application configuration for AutoML Web Platform."""

import os


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(32))
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    GENERATED_IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'generated')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS = {'csv'}
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MIN_ROWS = 10
