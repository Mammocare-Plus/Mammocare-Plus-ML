import os

CLASS_COUNT = 2

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../baseline_model.h5")

CLASS_NAMES = [
    'no cancer',
    'IDC (+)',
]