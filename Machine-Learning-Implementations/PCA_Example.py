import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
print(data.keys())

# Check the output classes
print(data['target_names'])

# Check the input attributes
print(data['feature_names'])

# construct a dataframe using pandas
df1 = pd.DataFrame(data['data'], columns=data['feature_names'])

# Scale data before applying PCA
scaling = StandardScaler()

# Use fit and transform method
scaling.fit(df1)
Scaled_data = scaling.transform(df1)

# Set the n_components=3
principal = PCA(n_components=3)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print('X Shape , after we reduce the features from 30 to 3', x.shape)

# Check the values of eigen vectors
# prodeced by principal components
principal.components_
plt.figure(figsize=(10, 10))
plt.scatter(x[:, 0], x[:, 1], c=data['target'], cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

# import relevant libraries for 3d graph

fig = plt.figure(figsize=(10, 10))

# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')

# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:, 0], x[:, 1], x[:, 2], c=data['target'], cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
plt.show()

# check how much variance is explained by each principal component
print(principal.explained_variance_ratio_)
