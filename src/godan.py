import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Load the dataset, skipping the first row
for dirname, _, filenames in os.walk('../docs/Crop_recommendation.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("success")
df = pd.read_excel('../docs/fadhili1(2).xlsx', skiprows=1)

# Handle summary statistics
print("Summary Statistics:")
print(df.describe())

# Handle missing values
df = df.dropna()
print("Shape after dropping missing values:", df.shape)

# Identify key features and distributions
print("Data Columns:")
print(df.columns)
print(df.isnull().sum())
print(df.head())

