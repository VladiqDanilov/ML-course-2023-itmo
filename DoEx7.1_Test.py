import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

df = pd.read_csv('persons_pics_train.csv')

print('1.1:', len(df['label'].unique()))
print('1.2:', df['label'].value_counts()['Hugo Chavez'] / df['label'].value_counts().sum())

mean_persons = df.groupby('label').mean()
print('1.3:', mean_persons.loc['George W Bush'][0])

#Картиночки с леблами из "1.4"
# fig, axs = plt.subplots(3, 4)

# for k, label in enumerate(df['label'].unique()):
#     axs[k//4][k%4].set_title(label)

#     img = []
#     for i in range(62):
#         img.append([])
#         for j in range(47):
#             img[i].append(mean_persons.loc[label][i*47+j])

#     axs[k//4][k%4].imshow(img, cmap='gray', vmin=0, vmax=1)

# plt.show()

Fname='Ariel Sharon'
Sname='Tony Blair'
ab = mean_persons.loc[[Fname, Sname]].prod(axis=0).sum()
a = sqrt(mean_persons.loc[[Fname, Fname]].prod(axis=0).sum())
b = sqrt(mean_persons.loc[[Sname, Sname]].prod(axis=0).sum())
print('1.5:', ab / (a * b))

x = df.drop('label', axis=1)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17, stratify=y)
svc = SVC(kernel='linear', random_state=17)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print('2.1:', f1_score(y_test, y_pred, average='weighted'))