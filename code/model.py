import csv
import pandas as pd
import numpy as np1
from sklearn import linear_model




df = pd.read_csv('Data.csv')
with open('Data.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    constants = [0,-0.74013,-8.68092,0.88240,0.21197,2.84211,0.53520,-0.21003,8.20833]
    feildnames = list(df.columns.values.tolist())

    clf = linear_model.LinearRegression()
    # clf.fit([[getattr(t, 'x%d' % i) for i in range(6, 14)] for t in df[feildnames]],
    #     [t.y for t in df['Sleep hours']])
    delta = []
    i=0
    sleep = constants[0]
    for constant in constants[1:]:
        sleep += constant* df[feildnames[i+6]]
        i = i+1
        print(i)
    y1 = df['Sleep hours']**2
    ym = sleep**2
    error = (df['Sleep hours'] - sleep)**2
    print(np1.sum(error))


# def fit(constants,)

    