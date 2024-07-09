from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()
"""I stored the features of the iris dataset into a variable called X.
Because it is an unsupervised learning task we should not teach the label or target"""

X = iris.data

#KMean clustering model.
kmeans = KMeans(n_clusters=3, random_state=56)
kmeans.fit(X)
#created a data frame using the features.
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(iris_df)
print(kmeans.labels_) #all the values based on which the features are clustered by the KMeans model

#included the labels in the iris data frame
iris_df['cluster'] = kmeans.labels_
print(iris_df)


#Plot
sns.scatterplot(x=iris_df['sepal length (cm)'], y=iris_df['sepal width (cm)'], hue=iris_df['cluster'], palette='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Mean Clustering of Iris Dataset')
plt.show()
