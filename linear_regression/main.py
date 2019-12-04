import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


df = pd.read_csv('./weather_lab.csv')
print(df)

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values
print(X)


