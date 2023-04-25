import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

f=open("Data3.1.CSV", "r")

dataSet1=pd.read_csv('Data3.1.CSV', delimiter=',', index_col='id')
# print(dataSet1)

res1=dataSet1.X.mean()
res2=dataSet1.Y.mean()


reg=LinearRegression().fit(dataSet1[['X']], dataSet1[['Y']])
res3=reg.intercept_[0]
res4=reg.coef_[0][0]
#res5=r2_score(dataSet1.Y, y)

R2=1-sum((dataSet1.Y-reg.intercept_-reg.coef_[0]*dataSet1.X)**2)/sum((dataSet1.Y-dataSet1.Y.mean())**2)
res5=R2
print('\n')
print(res1)
print(res2)
print('\n')
print("teta0: ", round(res3, 2))
print("teta1: ", round(res4, 2))
print("accur R**2: ", round(res5, 2))

