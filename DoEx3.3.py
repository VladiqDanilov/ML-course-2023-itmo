from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd

df_train = pd.read_csv('Data3.2.CSV')
df_test = pd.read_csv('Data3_3.csv')
#pca = PCA(n_components=1, svd_solver='full')
#pca.fit(df_train[['Length1', 'Length2', 'Length3']])

def clean(df):
    #df[['Width', 'Height', 'Lengths']] = df[['Width', 'Height', 'Lengths']].apply(lambda x: x**3)
    df[['Width', 'Height', 'Length1', 'Length2', 'Length3']] = df[['Width', 'Height', 'Length1', 'Length2', 'Length3']].apply(lambda x: x**2)
    dummies = pd.get_dummies(df['Species'], drop_first=True)
      
    df[list(dummies.columns)] = dummies
      
    print('dummies.columns: ', dummies.columns)
    df.drop(['Species'], axis=1, inplace=True)
      
clean(df_train)
clean(df_test)



x_train = df_train.drop(['Weight'], axis=1)
# print('1:')
# print(x_train)
# print('1222:')
# print(df_train)
y_train = df_train['Weight']
# print('2:')
x_test = df_test

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print(list(y_pred))