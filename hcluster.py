import clusters

row_names, column_names, data = clusters.readfile('dataset_vectors.txt')

clust = clusters.hcluster(data)
print('clusters by euclidean distance')
clusters.printhclust(clust, labels=row_names)
clusters.drawdendrogram(clust, row_names, jpeg='hcluster_euclidean_centroid.jpg')

print()
clust = clusters.hcluster(data, clusters.find_by_min)
print('clusters by euclidean distance')
clusters.printhclust(clust, labels=row_names)
clusters.drawdendrogram(clust, row_names, jpeg='hcluster_euclidean_min.jpg')

print()
clust = clusters.hcluster(data, clusters.find_by_max)
print('clusters by euclidean distance')
clusters.printhclust(clust, labels=row_names)
clusters.drawdendrogram(clust, row_names, jpeg='hcluster_euclidean_max.jpg')