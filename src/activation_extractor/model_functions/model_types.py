"""
This script reads the model_types from model_types.csv . 
"""

import importlib.resources as pkg_resources
import pandas as pd

with pkg_resources.open_text('activation_extractor.model_functions', 'model_types.csv') as csv_file:
    df = pd.read_csv(csv_file)
    df = df.dropna(subset="model_type")

model_types = df