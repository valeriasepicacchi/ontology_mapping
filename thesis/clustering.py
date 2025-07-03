from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_scorex
import numpy as np
''' These experiments were run on a company owned dataset to produce a clustering and are therefore not reproducible'''
# Read the training set from CSV 
training_set = pd.read_csv("taxonomy_training_set.csv")
print(training_set.columns)
training_set["Text"] = training_set["Preferred Label"]# + " " + training_set["Alternate Labels"].fillna("")

# TF-IDF Vectorization to convert text into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(training_set["Text"])

print("TF-IDF Matrix Shape:", X.shape)
n_clusters = 6  

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.fit_predict(X)
silhouette = silhouette_score(X,labels )
print('silhouette score is: ',silhouette)
#davies_score = davies_bouldin_score(X, labels)
#print('davies bouldin score is: ', davies_score)
# print('done')
# Assign cluster labels to each concept
training_set["Cluster"] = kmeans.labels_

# Show the clustered concepts
print(training_set[["Preferred Label", "Cluster"]].head())
terms = np.array(vectorizer.get_feature_names_out())

# For each cluster, print the top terms
for i in range(n_clusters):
    print(f"Cluster {i} Top Terms:")
    cluster_center = kmeans.cluster_centers_[i]
    top_terms_idx = cluster_center.argsort()[-10:][::-1]
    top_terms = terms[top_terms_idx]
    print(" ".join(top_terms))
    print("-" * 50)


# Reduce dimensions using t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())

#Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_, cmap="viridis", marker="o")
plt.colorbar()
plt.title("t-SNE Clustering of Taxonomy Concepts")
plt.show()
training_set.to_csv("taxonomy_clustered_data.csv", index=False)
print("Clustered data saved to taxonomy_clustered_data.csv")
