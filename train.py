# coding: utf-8
import pandas as pd
from sklearn.svm import SVC
df = pd.read_csv("data.csv")
y = df.iloc[:,-1]
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
mysvm = SVC().fit(df,y)
