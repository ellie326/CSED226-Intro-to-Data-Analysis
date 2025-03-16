import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


df_o = pd.read_csv('train.csv')

df = df_o.drop('ID', axis=1)

#df.describe()

pca = PCA() 
df_pca = pca.fit_transform(df)
explained_variance = pca.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance)
n_components = np.where(cum_explained_variance > 0.95)[0][0] + 1
#print(n_components)


pca = PCA(n_components=10) 
df_pca = pca.fit_transform(df)
columns = ['PC'+str(i+1) for i in range(10)] 
df_pca = pd.DataFrame(df_pca, columns=columns)
#df_pca.describe()


Q1 = df_pca.quantile(0.25)
Q3 = df_pca.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((df_pca < (Q1 - 1.5 * IQR)) | (df_pca > (Q3 + 1.5 * IQR)))
df_outlier_replaced = df_pca.mask(outlier_mask, df_pca.mean(), axis=1)

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_outlier_replaced), columns=df_outlier_replaced.columns)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)


'''K = range(2,30)
distortions = []
silhouettes = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df_scaled)
    distortions.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(df_scaled, kmeans.labels_))

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

plt.figure(figsize=(16,8))
plt.plot(K, silhouettes, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('The Silhouette Method showing the optimal k')
plt.show()'''

optimal_k = 3 # 엘보우에서의 k 값
kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(df_scaled)
labels = kmeans.labels_
submission = pd.DataFrame({'ID': df_scaled.index, 'label': labels})
submission.to_csv('submission15.csv', index=False)