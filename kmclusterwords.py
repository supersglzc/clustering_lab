import clusters

docs,words,data=clusters.readfile('titles_vectors.txt')

num_clusters=2
print('Searching for {} clusters:'.format(num_clusters))
print()
clust=clusters.kcluster(data,distance=clusters.pearson,k=num_clusters)
print ('clusters by pearson correlation')
for i in range(num_clusters):
    print ('cluster {}:'.format(i+1))
    print ([words[r] for r in clust[i]])

print()
clust=clusters.kcluster(data,distance=clusters.tanimoto,k=num_clusters)
print ('clusters by tanimoto coefficient')
for i in range(num_clusters):
    print ('cluster {}:'.format(i+1))
    print ([words[r] for r in clust[i]])

print()
clust=clusters.kcluster(data,distance=clusters.euclidean,k=num_clusters)
print ('clusters by euclidean distance')
for i in range(num_clusters):
    print ('cluster {}:'.format(i+1))
    print ([words[r] for r in clust[i]])

print()
clust=clusters.kcluster(data,distance=clusters.cosine,k=num_clusters)
print ('clusters by cosine distance')
for i in range(num_clusters):
    print ('cluster {}:'.format(i+1))
    print ([words[r] for r in clust[i]])
