from cProfile import label
import csv
from turtle import color, title
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

file_name= "full-data.csv"


lr = LinearRegression()

def plotgraph(X,y, name, pred):
    name = name.capitalize()
    title = "Linear Regression for " + name + " vs Sleep"
    plt.style.use('fivethirtyeight')
    plt.grid(False)
    plt.title(title)
    plt.ylabel('Sleep')
    # plot with increment of 1 for x axis
    if name == "year standing":
        plt.plot(X, y, 'o')
        plt.plot(X, pred)
        plt.xlabel('Year Standing')
        
        plt.show()
    else:
        plt.figure()
        plt.scatter(X,y, color='#495867', linewidth=1, label = "Actual")
        plt.plot(X, pred, color='#FE5F55', linewidth=1, label = "Best fit")
        plt.xlabel(name)

    plt.legend()
    plt.show()

def fitdata(X,y, name):
    samples = X.shape[0]
    dof = X.shape[1]
    lr.fit(X, y)
    c = lr.intercept_
    m = lr.coef_
    r2 = r2_score(y, lr.predict(X))
    adjr2 = 1 - (1 - r2) * (samples - 1) / (samples - dof - 1)
    if name != "All":
        plotgraph(X,y, name, lr.predict(X))
    # else:
    #     # write values of column and c[i] to csv
    #     with open('readings.csv', 'w', encoding='UTF8') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Column name", "Intercept", "Coefficients", "R2", "Adj R2"])
    #         for i in range(len(X.columns)):
    #             writer.writerow([X.columns[i], m[i]])
    if len(m) == 1:
       m=m[0]
    return [c, m, r2, adjr2]


df = pd.read_csv(file_name)
cat_variables = ["gender", "ethnicity", "major"]


# for st in cat_variables:
#     print(st)
#     print(df[st])
#     df[st] = df[st].astype("category")
#     df[st] = df[st].cat.codes
#     print(df[st])





