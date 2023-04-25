from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

task2 = pd.read_csv('Data4_2.csv')


to_delete = ['education', 'marital-status']
task2.drop(to_delete, axis=1, inplace=True)
print('2.1:')

cnt=0
for col in task2.columns:
    if task2.dtypes.loc[col] == 'object':
        cnt+=1
print(cnt)
print('2.2:', task2['label'].value_counts().loc[0] / len(task2))


non_numeric_columns = [col for col in task2.columns if task2.dtypes.loc[col] == 'object']
task2_clean = task2.drop(non_numeric_columns, axis=1)

x_train, x_test, y_train, y_test = train_test_split(task2_clean.drop('label', axis=1), task2_clean['label'],
    test_size=0.2, random_state=7, stratify=task2_clean['label'])
print('2.3:', x_train['fnlwgt'].mean())
knn1 = KNeighborsClassifier()
knn1.fit(x_train, y_train)
y_pred = knn1.predict(x_test)
print('2.4:', f1_score(y_test, y_pred))



scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

print('2.5:', x_train['fnlwgt'].mean())

knn2 = KNeighborsClassifier()
knn2.fit(x_train, y_train)
y_pred = knn2.predict(x_test)

print('2.6:', f1_score(y_test, y_pred))

print('Картиночки 2.7')

fig, axs = plt.subplots(1, len(non_numeric_columns))

for ax, col in zip(axs, non_numeric_columns):
    c = task2[col].value_counts()
    ax.bar(c.index, c)
    ax.set_title(col)

plt.show()

rows_with_nan = task2.apply(lambda row: (row=='?').any(), axis=1)
print('2.8:', rows_with_nan.sum())
task2_with_dummies = task2[~rows_with_nan].copy()

for col in non_numeric_columns:
    dummies = pd.get_dummies(task2_with_dummies[col], drop_first=True)
    task2_with_dummies[dummies.columns] = dummies
    task2_with_dummies.drop(col, axis=1, inplace=True)


print('2.9:', len(task2_with_dummies.columns) - 1)








x_train, x_test, y_train, y_test = train_test_split(task2_with_dummies.drop('label', axis=1),
    task2_with_dummies['label'], test_size=0.2, random_state=7, stratify=task2_with_dummies['label'])

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

knn3 = KNeighborsClassifier()
knn3.fit(x_train, y_train)
y_pred = knn3.predict(x_test)
print('2.10:', f1_score(y_test, y_pred))

for col in task2.columns:
    task2[col].replace('?', task2[col].mode().values[0], inplace=True)

for col in non_numeric_columns:
    dummies = pd.get_dummies(task2[col], drop_first=True)
    task2[dummies.columns] = dummies
    task2.drop(col, axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(task2.drop('label', axis=1), task2['label'], 
    test_size=0.2, random_state=7,stratify=task2['label'])

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

knn4 = KNeighborsClassifier()
knn4.fit(x_train, y_train)
y_pred = knn4.predict(x_test)
print('2.11:', f1_score(y_test, y_pred))