"""Sample datasets configuration."""

SAMPLE_DATASETS = [
    {
        'id': 'breast-cancer',
        'name': 'Breast Cancer',
        'icon': '🏥',
        'task_type': 'Classification',
        'rows': 569,
        'columns': 32,
        'target': 'diagnosis',
        'description': 'Predict if tumor is malignant (M) or benign (B)',
        'file': 'breast-cancer-data.csv',
    },
    {
        'id': 'cereal',
        'name': 'Cereal Nutrition',
        'icon': '🥣',
        'task_type': 'Regression',
        'rows': 77,
        'columns': 16,
        'target': 'rating',
        'description': 'Predict cereal health rating from nutritional data',
        'file': 'cereal.csv',
    },
]
