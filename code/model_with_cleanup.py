import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


lr = LinearRegression()

def fitdata(X,y):
    samples = X.shape[0]
    dof = X.shape[1]
    lr.fit(X, y)
    c = lr.intercept_
    m = lr.coef_
    r2 = r2_score(y, lr.predict(X))
    adjr2 = 1 - (1 - r2) * (samples - 1) / (samples - dof - 1)
    if len(m) == 1:
       m=m[0]
    return [c, m, r2, adjr2]


df = pd.read_csv("full-data.csv")
cat_variables = ["year standing", "caffeine", "work", "social", "eca", "nap", "sex"]


for st in cat_variables:
    df[st] = df[st].astype("category")
    df[st] = df[st].cat.codes

if df.isnull().sum().sum() != 0:
    raise Exception("Data contains null values. Can't proceed with linear regression")

X = df.drop(columns=["sleep"])
y = df["sleep"]




# print(fitdata(X,y))
with open('readings.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(["Column name", "Intercept", "Coefficients", "R2", "Adj R2"])

    for column in X.columns:
        row= [column] + fitdata(X[column].values.reshape(-1,1), y)
        plt.figure()
        plt.plot(X,y, label='left')
        writer.writerow(row)


