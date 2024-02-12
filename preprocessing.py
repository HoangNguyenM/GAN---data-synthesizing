import pandas as pd
import numpy as np


df = pd.read_csv("data/heart.csv")

origin_class = df['DEATH']

X = df.drop('DEATH',axis=1)
col = X.columns

for c in col:
    temp = X[c].to_numpy()
    temp = temp.astype(float)
    if np.min(temp) != np.max(temp):
        temp -= np.min(temp)
        temp -= np.max(temp)/2
        temp /= np.max(temp)
    X[c] = temp
    
X['DEATH'] = origin_class

X.to_csv("data/heart_train.csv")
