import clusters
from math import *


def sse_error(data, centers, cluster, numbers):
    error = 0
    for element in range(numbers):
        for m in range(len(cluster[element])):
            for j in range(len(centers[element])):
                error += (data[cluster[element][m]][j] - centers[element][j]) ** 2
    return error


def bisect(data, number):
    cluster_list = []
    centers_list = []
    count = 0
    while count != number:
        if count == 0:
            cluster, centroid = clusters.kcluster(data, distance=clusters.euclidean, k=2)
            cluster_list.append(cluster[0])
            cluster_list.append(cluster[1])
            centers_list.append(centroid[0])
            centers_list.append(centroid[1])
            count += 2
        else:
            max_error = 0
            index = 0
            for f in range(len(cluster_list)):
                error = 0
                for j in range(len(cluster_list[f])):
                    for m in range(len(centers_list[f])):
                        error += (data[cluster_list[f][j]][m] - centers_list[f][m]) ** 2
                if max_error < error:
                    max_error = error
                    index = f
            new_data = []
            buffer = cluster_list[index]
            for k in buffer:
                new_data.append(data[k])
            cluster_list.pop(index)
            centers_list.pop(index)
            cluster, centroid = clusters.kcluster(new_data, distance=clusters.euclidean, k=2)
            for n in cluster:
                new = []
                for l in n:
                    new.append(buffer[l])
                cluster_list.append(new)

            centers_list.append(centroid[0])
            centers_list.append(centroid[1])
            count += 1
        # print(cluster_list)
    return cluster_list, centers_list


def convert_string(first, name, group):
    line = ""
    if first == 0:
        line += ', '
    if name == "Arab countries":
        countries = ["Bahrain", "Comoros", "Djibouti", "Kuwait", "Lebanon", "Libya", "Mauritania", "Oman", "Palestine",
                     "Qatar", "Somalia", "Sudan", "Syria", "Tunisia", "United Arab Emirates", "Yemen"]
        for country in countries[:-1]:
            line += '{"Country": "' + country + '", "Cluster": ' + str(group) + '}, '
        line += '{"Country": "' + countries[-1] + '", "Cluster": ' + str(group) + '}'
    elif name == "Africa West":
        countries = ["Benin", "Cabo Verde", "Ivory Coast", "Gambia", "Guinea", "Guinea-Bissau", "Liberia", "Niger",
                     "Senegal", "Sierra Leone", "Togo"]
        for country in countries[:-1]:
            line += '{"Country": "' + country + '", "Cluster": ' + str(group) + '}, '
        line += '{"Country": "' + countries[-1] + '", "Cluster": ' + str(group) + '}'
    elif name == "Africa East":
        countries = ["Eritrea", "Ethiopia", "South Sudan", "Madagascar", "Mauritius", "Seychelles", "Reunion",
                     "Mayotte", "Burundi", "Kenya", "Malawi", "Mozambique"]
        for country in countries[:-1]:
            line += '{"Country": "' + country + '", "Cluster": ' + str(group) + '}, '
        line += '{"Country": "' + countries[-1] + '", "Cluster": ' + str(group) + '}'
    elif name == "Czech Rep":
        line += '{"Country": "' + "Czech Republic" + '", "Cluster": ' + str(group) + '}'
    elif name == "Dominican Rep":
        line += '{"Country": "' + "Dominican Republic" + '", "Cluster": ' + str(group) + '}'
    elif name == "Kyrgyz Rep":
        line += '{"Country": "' + "Kyrgyzstan" + '", "Cluster": ' + str(group) + '}'
    elif name == "Korea South":
        line += '{"Country": "' + "South Korea" + '", "Cluster": ' + str(group) + '}'
    elif name == "Macedonia Rep":
        line += '{"Country": "' + "Macedonia" + '", "Cluster": ' + str(group) + '}'
    elif name == "U.S.A.":
        line += '{"Country": "' + "United States" + '", "Cluster": ' + str(group) + '}'
    elif name == "Slovak Rep":
        line += '{"Country": "' + "Slovakia" + '", "Cluster": ' + str(group) + '}'
    else:
        line += '{"Country": "' + name + '", "Cluster": ' + str(group) + '}'
    return line


row_names, column_names, data = clusters.readfile('dataset_vectors.txt')
# rdata = clusters.rotatematrix(data)
num_clusters = 6
print('Grouping countries into {} clusters:'.format(num_clusters))
# print(rdata)
print()
clust, centers = clusters.kcluster(data, distance=clusters.pearson, k=num_clusters)
print('clusters by pearson correlation')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print([row_names[r] for r in clust[i]])
print("The SSE Error of pearson correlation is: ", sse_error(data, centers, clust, num_clusters))

print()
clust, centers = clusters.kcluster(data, distance=clusters.euclidean, k=num_clusters)
print('clusters by euclidean distance')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print([row_names[r] for r in clust[i]])
print("The SSE Error of euclidean is: ", sse_error(data, centers, clust, num_clusters))

print()
clust, centers = clusters.kcluster(data, distance=clusters.cosine, k=num_clusters)
print('clusters by cosine distance')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print([row_names[r] for r in clust[i]])
print("The SSE Error of cosine distance is: ", sse_error(data, centers, clust, num_clusters))

print()
clust, centers = bisect(data, num_clusters)
print('clusters by bisecting K-means')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print([row_names[r] for r in clust[i]])
print("The SSE Error of bisecting K-means is: ", sse_error(data, centers, clust, num_clusters))

# print(clust)
# print(centers)

# rank the clusters according to their centroid distances
final_list = [clust[0]]
buffer_dic = {}
buffer_list = []
for i in range(len(centers[1:])):
    buffer = 0
    for j in range(len(centers[i])):
        buffer += sqrt((centers[0][j] - centers[i + 1][j]) ** 2)
    buffer_dic[buffer] = i + 1
    buffer_list.append(buffer)

buffer_list.sort()

for i in buffer_list:
    final_list.append(clust[buffer_dic[i]])


# write the data to data.js file
first_line = 1
file1 = open('./public_html/data.js', 'w', newline='')
file1.write("var data1 = [")
group_id = 0
for i in final_list:
    for j in i:
        if first_line:
            file1.write(convert_string(first_line, row_names[j], group_id))
            first_line = 0
        else:
            file1.write(convert_string(first_line, row_names[j], group_id))
    group_id += 1
file1.write('];')
file1.close()

clusters.descriptive_label(data, clust)
