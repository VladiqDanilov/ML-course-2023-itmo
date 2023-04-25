from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/9217064c15d227845c04eca083c04252/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/37_25.csv', header=None)

# X_reduced1 = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/22a15a7ec79aa6b658e43ee21d89c935/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/X_reduced_513.csv',
#                         header=None, sep=';').values.astype(float)
# X_loader = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/ccd8b40cd2f55b4c899db64587b05266/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/X_loadings_513.csv',
#                          header=None, sep=';').values.astype(float)



print("data type:", type(data))
pca = PCA(n_components=2)
pca.fit(data)


new_coordinates = pca.transform(data)
x1, y1 = new_coordinates[0]
print(new_coordinates[0])



variance_ratio = sum(pca.explained_variance_ratio_[:2])
res=pca.explained_variance_ratio_
print(sum(res))
#print(res)

n_components = PCA(n_components=0.85).fit(data).n_components_


kmeans = KMeans(n_clusters=5, n_init = 10)
kmeans.fit(new_coordinates)
value=len(set(kmeans.labels_))
val1=len(set(kmeans.labels_))

print(f"первой гл комп: {x1:.3f}, относительно второй {y1:.3f}")
print(f"объед дисп {variance_ratio:.3f}")

print("мин колличество ГК", n_components)
print("колличество групп при двух компонентах: ", val1)



print("\nЗадание 2\n")


X_reduced = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/71a87f8f051e8169e8e90dd0d57d2644/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/X_reduced_681.csv',
                        header=None, sep=';').values.astype(float)
X_loadings = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/86cb6fbc75f48622dd4c3466aaed034c/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/X_loadings_681.csv',
                         header=None, sep=';').values.astype(float)


# вычисление произведения матриц
X_restored = np.dot(X_reduced, X_loadings.T)

# приведение значений к диапазону от 0 до 255
X_restored = (X_restored - X_restored.min()) / \
    (X_restored.max() - X_restored.min()) * 255

# преобразование матрицы в изображение и сохранение его в файл
img = Image.fromarray(X_restored.astype('uint8'))
img.save('restored_image.png')
print("Сопоставьте фото с номером")

from IPython.display import Image
Image(filename='restored_image.png')

