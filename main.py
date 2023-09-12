import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

shap.initjs()

data = pd.read_csv(
    filepath_or_buffer='./data/abalone.data',
    names=[
        "sex",
        "length",
        "diameter",
        "height",
        "whole weight",
        "shucked weight",
        "viscera weight",
        "shell weight",
        "rings",
    ]
)

print(len(data))
print(data.head())
