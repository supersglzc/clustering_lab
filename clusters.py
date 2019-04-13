from PIL import Image, ImageDraw
from math import *
import random
from pytagcloud import create_tag_image, make_tags
import csv


def readfile(file_name):
    f = open(file_name)
    lines = [line for line in f]
  
    # First line is the column titles
    colnames = lines[0].strip().split('\t')[2:]
    # print(colnames)
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        # First column in each row is the rowname
        if len(p) > 1:
            rownames.append(p[2])
            # The data for this row is the remainder of the row
            data.append([float(x) for x in p[3:]])
    # print(data)
    # print(rownames)
    return rownames, colnames, data


def rotatematrix(data):
    newdata = []
    for i in range(len(data[0])):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata


def print_2d_array(matrix):
    for i in range(len(matrix)):
        for j in range (len(matrix[i])):
            print (matrix[i][j], end="")
        print('\n')


# different similarity metrics for 2 vectors
def manhattan(v1,v2):
    res = 0
    dimensions = min(len(v1),len(v2))

    for i in range(dimensions):
        res += abs(v1[i]-v2[i])

    return res


def euclidean(v1, v2):
    res = 0
    dimensions = min(len(v1), len(v2))
    for i in range(dimensions):
        res += pow(abs(v1[i]-v2[i]), 2)

    return sqrt(float(res))


def cosine(v1, v2):
    dotproduct = 0
    dimensions = min(len(v1), len(v2))

    for i in range(dimensions):
        dotproduct += v1[i]*v2[i]

    v1len = 0
    v2len = 0
    for i in range(dimensions):
        v1len += v1[i]*v1[i]
        v2len += v2[i]*v2[i]

    v1len = sqrt(v1len)
    v2len = sqrt(v2len)

    return 1.0-(float(dotproduct)/(v1len*v2len))
  

def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
  
    # Sums of the squares
    sum1Sq = sum([pow(v, 2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])
  
    # Sum of the products
    pSum = sum([v1[i]*v2[i] for i in range(min(len(v1), len(v2)))])
  
    # Calculate r (Pearson score)
    num = pSum-(sum1*sum2/len(v1))
    den = sqrt((sum1Sq-pow(sum1, 2)/len(v1))*(sum2Sq-pow(sum2, 2)/len(v1)))
    if den == 0:
        return 1.0
  
    return 1.0-num/den


def tanimoto(v1, v2):
    c1, c2, shr = 0, 0, 0

    for i in range(len(v1)):
        if v1[i] != 0:
            c1 += 1 # in v1
        if v2[i] != 0:
            c2 += 1 # in v2
        if v1[i] != 0 and v2[i] != 0:
            shr += 1 # in both

    return 1.0-(float(shr)/(c1+c2-shr))


# Hierarchical clustering
class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance


def find_by_centroid(clust, currentclustid, distance=euclidean):
    while len(clust) > 1:
        lowest_pair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                d = distance(clust[i].vec, clust[j].vec)

                if d < closest:
                    closest = d
                    lowest_pair = (i, j)

        mergevec = [(clust[lowest_pair[0]].vec[i] + clust[lowest_pair[1]].vec[i]) / 2.0 for i in range(len(clust[0].vec))]

        newcluster = bicluster(mergevec, left=clust[lowest_pair[0]], right=clust[lowest_pair[1]], distance=closest,
                               id=currentclustid)

        del clust[lowest_pair[1]]
        del clust[lowest_pair[0]]
        clust.append(newcluster)
        currentclustid -= 1

    return clust[0]


def get_vectors(bicluster):
    if (bicluster.right is None) and (bicluster.left is None):
        return [bicluster.vec]
    if (bicluster.right is None) and (bicluster.left is not None):
        return get_vectors(bicluster.left)
    if (bicluster.right is not None) and (bicluster.left is None):
        return get_vectors(bicluster.right)
    else:
        return get_vectors(bicluster.right) + get_vectors(bicluster.left)


def find_by_min(clust, currentclustid, distance=euclidean):
    while len(clust) > 1:
        lowest_pair = (0, 1)
        closet = distance(clust[0].vec, clust[1].vec)
        for i in range(len(clust)):
            result1 = get_vectors(clust[i])
            # print(result1)
            for j in range(i + 1, len(clust)):
                result2 = get_vectors(clust[j])
                dis = 1000000000000
                for m in range(len(result1)):
                    for n in range(len(result2)):
                        buffer = distance(result1[m], result2[n])
                        if buffer < dis:
                            dis = buffer
                if dis < closet:
                    lowest_pair = (i, j)

        mergevec = [(clust[lowest_pair[0]].vec[i] + clust[lowest_pair[1]].vec[i]) / 2.0 for i in
                    range(len(clust[0].vec))]

        newcluster = bicluster(mergevec, left=clust[lowest_pair[0]], right=clust[lowest_pair[1]], distance=closet,
                               id=currentclustid)

        del clust[lowest_pair[1]]
        del clust[lowest_pair[0]]
        clust.append(newcluster)
        currentclustid -= 1
    return clust[0]


def find_by_max(clust, currentclustid, distance=euclidean):
    while len(clust) > 1:
        lowest_pair = (0, 1)
        closet = distance(clust[0].vec, clust[1].vec)
        for i in range(len(clust)):
            result1 = get_vectors(clust[i])
            # print(result1)
            for j in range(i + 1, len(clust)):
                result2 = get_vectors(clust[j])
                dis = 0
                for m in range(len(result1)):
                    for n in range(len(result2)):
                        buffer = distance(result1[m], result2[n])
                        if buffer > dis:
                            dis = buffer
                if dis < closet:
                    lowest_pair = (i, j)

        mergevec = [(clust[lowest_pair[0]].vec[i] + clust[lowest_pair[1]].vec[i]) / 2.0 for i in
                    range(len(clust[0].vec))]

        newcluster = bicluster(mergevec, left=clust[lowest_pair[0]], right=clust[lowest_pair[1]], distance=closet,
                               id=currentclustid)

        del clust[lowest_pair[1]]
        del clust[lowest_pair[0]]
        clust.append(newcluster)
        currentclustid -= 1
    return clust[0]


def hcluster(rows, method=find_by_centroid):
    currentclustid = -1

    # Clusters are initially just the rows
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
    # print(rows)

    return method(clust, currentclustid)


def printhclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n):
        print (' ', end="")
    if clust.id<0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels is None:
            print(clust.id)
        else:
            print(labels[clust.id])

    # now print the right and left branches
    if clust.left!=None: printhclust(clust.left, labels=labels, n=n+1)
    if clust.right!=None: printhclust(clust.right, labels=labels, n=n+1)


# draw hierarchical clusters
def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left==None and clust.right==None: return 1

    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left)+getheight(clust.right)


def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left==None and clust.right==None: return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance


def drawdendrogram(clust,labels,jpeg='clusters.jpg'):
    # height and width
    h=getheight(clust)*20
    w=1200
    depth=getdepth(clust)

    # width is fixed, so scale distances accordingly
    scaling=float(w-150)/depth

    # Create a new image with a white background
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    draw.line((0,h/2,10,h/2),fill=(255,0,0))

    # Draw the first node
    drawnode(draw,clust,10,(h/2),scaling,labels)
    img.save(jpeg,'JPEG')


def drawnode(draw,clust,x,y,scaling,labels):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # Line length
        ll=clust.distance*scaling
        # Vertical line from this cluster to children
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))

        # Horizontal line to left item
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))

        # Horizontal line to right item
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))

        # Call the function to draw the left and right nodes
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
    else:
        # If this is an endpoint, draw the item label
        draw.text((x+5,y-7),labels[clust.id],(0,0,0))


# k-means clustering
def kcluster(rows, distance=euclidean, k=4):
    # Determine the minimum and maximum values for each point
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
    for i in range(len(rows[0]))]
    # Create k randomly placed centroids
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
    for i in range(len(rows[0]))] for j in range(k)]
  
    lastmatches = None
    bestmatches = None

    for t in range(100):
        # print('Iteration %d' % t)
        bestmatches = [[] for i in range(k)]
    
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row = rows[j]
            bestmatch=0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row): bestmatch = i
            bestmatches[bestmatch].append(j)

        # If the results are the same as last time, this is complete
        if bestmatches == lastmatches:
            # print(clusters)
            # print(bestmatches)
            break
        lastmatches = bestmatches

        # Move the centroids to the average of their members
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs
      
    return bestmatches, clusters


def scaledown(data,distance=pearson,rate=0.01):
    n=len(data)

    # The real distances between every pair of items
    realdist=[[distance(data[i],data[j]) for j in range(n)]
             for i in range(0,n)]

    # Randomly initialize the starting points of the locations in 2D
    loc=[[random.random(),random.random()] for i in range(n)]
    fakedist = [[0.0 for j in range(n)] for i in range(n)]
  
    lasterror = None
    for m in range(0, 1000):
        # Find projected distances
        for i in range(n):
            for j in range(n):
                fakedist[i][j] = sqrt(sum([pow(loc[i][x]-loc[j][x], 2)
                                 for x in range(len(loc[i]))]))
  
        # Move points
        grad = [[0.0, 0.0] for i in range(n)]
    
        totalerror = 0
        for k in range(n):
            for j in range(n):
                if j == k: continue
                # The error is percent difference between the distances
                errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k]
        
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k])*errorterm
                grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm

                # Keep track of the total error
                totalerror+=abs(errorterm)
        print ("Total error:",totalerror)

        # If the answer got worse by moving the points, we are done
        if lasterror and lasterror<totalerror: break
        lasterror=totalerror
    
        # Move each of the points by the learning rate times the gradient
        for k in range(n):
            loc[k][0]-=rate*grad[k][0]
            loc[k][1]-=rate*grad[k][1]

    return loc


def draw2d(data,labels,jpeg='mds2d.jpg'):
    img = Image.new('RGB',(2000,2000),(255,255,255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0]+0.5)*1000
        y = (data[i][1]+0.5)*1000
        draw.text((x, y), labels[i], (0, 0, 0))
    img.save(jpeg, 'JPEG')


def create_cloud(oname, words, maxsize=60, fontname='Lobster'):
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, oname, size=(2000, 2000), fontname=fontname)


def descriptive_label(rows, clusters):
    print("Start to process descriptive labels ...")
    words = []
    first = True
    with open('dimensions_keywords.csv') as csv_file:
        file = csv.reader(csv_file, delimiter=',', quotechar='"')
        for i in file:
            if first:
                first = False
                continue
            list(i)
            words.append([i[1].split(), i[2].split()])

    count = 1
    for cluster in clusters:
        centers = [0] * 6
        number = 0
        word_dic = []
        for i in cluster:
            for j in range(6):
                centers[j] += rows[i][j]
            number += 1
        for i in range(6):
            centers[i] = int(centers[i] / number)
            for w in words[i][0]:
                word_dic.append((w, centers[i]))
            for w in words[i][1]:
                word_dic.append((w, 100 - centers[i]))
        create_cloud("word_cloud_cluster_" + str(count) + ".png", word_dic)
        count += 1
    print("Finish")
