import numpy as np
import csv
import pandas as pd
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from pylab import *
from matplotlib import style
import validation as var


#Load some data(csv file
#style.use("ggplot")
#pd_data=pd.read_csv('final.csv')
#convert the csv file into numpy_array
#numpy_data=np.array(pd_data)
#X=numpy_data
X=var.new_centroidlist
#set the number of cluster you want from the provided data
cluster_num = 1
#Run K-Means
kmeans = KMeans(n_clusters=cluster_num)
kmeans.fit(X)
#store the centroids
centroids = kmeans.cluster_centers_
#store the labels
labels = kmeans.labels_
fig = figure()
ax = fig.gca(projection='3d')
#set the color according to the number of the cluster
colors = ["r"]
ax.text2D(0.94,1.01, "0", color='red',transform=ax.transAxes)
#ax.text2D(0.96,1.01, "1", color='blue',transform=ax.transAxes)
#ax.text2D(0.98,1.01, "2", color='green',transform=ax.transAxes)
#ax.text2D(1.0,1.01, "3", color='yellow',transform=ax.transAxes)

color = np.random.rand(cluster_num)
c = Counter(labels)
for i in range(len(X)):
    #print("coordinate:",X[i])
    # print "i : ",i
    #print "color[labels[i]] : ",color[labels[i]]
    ax.scatter(X[i][0],X[i][1],X[i][2], c=colors[labels[i]])

k=0.05
l=0.95
p=0.94
#Plot cluster membership for each instance
for cluster_number in range(cluster_num):

 # print("person {} contains {} samples ".format(cluster_number, c[cluster_number]),centroids[cluster_number])
  ax.text2D(k, l, "{}-person {} contains {} samples".format(colors[cluster_number],cluster_number, c[cluster_number]), transform=ax.transAxes)
  l=l+0.04
ax.scatter(centroids[:, 0],centroids[:, 1],centroids[:, 2], marker ="p", s=150, linewidths = 5, zorder = 100, c=colors)
plt.show()
