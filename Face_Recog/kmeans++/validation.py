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
import kmean_3d as var

#set the 3d visulisation using Axes 3d
fig = figure()
ax = fig.gca(projection='3d')

style.use("ggplot")
pd_data=pd.read_csv('output3.csv')
#convert the csv file into numpy_array
numpy_data=np.array(pd_data)
Y=numpy_data

flag=3


#set the color according to the number of the cluster
colors = ["r","b","g","black"]
ax.text2D(0.94,1.01, "0", color='red',transform=ax.transAxes)
ax.text2D(0.96,1.01, "1", color='blue',transform=ax.transAxes)
ax.text2D(0.98,1.01, "2", color='green',transform=ax.transAxes)
#ax.text2D(1.0,1.01, "3", color='yellow',transform=ax.transAxes)


#list contating samples in cluster 0
p1=list()
#list contaning samples in cluster 1
p2=list()
p3=list()

#loop the actual data and label 0 store in list p1 and label 1 in list p2
for i in range(len(var.X)):
    if (var.labels[i] == 0):
        p1.append(var.X[i])
    elif(var.labels[i]==1):
        p2.append(var.X[i])
    else:
        p3.append(var.X[i])


k1=len(p1)
k2=len(p2)
k3=len(p3)
#finding the outlier in the cluster 0
outlier0 = abs(var.centroids[0]-p1[0])
for i in range(1,len(p1)):
    if(cmp(abs(var.centroids[0]-p1[i]).all,outlier0.all)):
        outlier0=abs(var.centroids[0]-p1[i])



#finding the outlier in the cluster1
outlier1 = abs(var.centroids[1]-p2[0])
for i in range(1,len(p2)):
    if(cmp(abs(var.centroids[1]-p2[i]).all,outlier1.all)):
        outlier1=abs(var.centroids[1]-p2[i])

outlier2=abs(var.centroids[2]-p3[0])
for i in range(1,len(p3)):
    if(cmp(abs(var.centroids[2]-p3[i]).all,outlier2.all)):
        outlier2=abs(var.centroids[2]-p3[i])


#calculating the mean of outlier0
mean_outlier0=np.mean(outlier0)
#calculating the mean of outlier1
mean_outlier1=np.mean(outlier1)
mean_outlier2=np.mean(outlier2)
#calculating the mean of centroid0
mean_centroid0=np.mean(var.centroids[0])
#calculating the mean of centroid1
mean_centroid1=np.mean(var.centroids[1])
mean_centroid2=np.mean(var.centroids[2])

lar0=abs(mean_outlier0-mean_centroid0)
radi0=lar0
lar1=abs(mean_outlier1-mean_centroid1)
radi1=lar1
lar2=abs(mean_outlier2-mean_centroid2)
radi2=lar2



new_centroidlist=list();
p0=0
p1=0
p2=0
for i in range(len(Y)):
    my_list=list()
    for j in range(len(var.centroids)):
        a=Y[i]-var.centroids[j]
        my_list.append(a)
    c1 = np.mean(my_list[2])
    d1 = np.mean(my_list[1])
    b1=np.mean(my_list[0])
    if(abs(c1)<abs(d1) and abs(c1)<abs(b1)):
        if(abs(c1)<lar0):
            p0=p0+1
            ax.scatter(Y[i][0], Y[i][1], Y[i][2], c=colors[0])
        else:
            new_centroidlist.append(Y[i])


    elif (abs(d1) < abs(c1) and abs(d1)<abs(b1)):
        if (abs(d1) < lar1):
            p1 = p1 + 1
            ax.scatter(Y[i][0], Y[i][1], Y[i][2], c=colors[1])
        else:
             new_centroidlist.append(Y[i])

    elif (abs(b1) < abs(c1) and abs(b1) < abs(d1)):
        if (abs(b1) < lar2):
            p2 = p2 + 1
            ax.scatter(Y[i][0], Y[i][1], Y[i][2], c=colors[2])
        else:
             new_centroidlist.append(Y[i])


ax.scatter(var.centroids[:, 0],var.centroids[:, 1],var.centroids[:, 2], marker ="p", s=150, linewidths = 5, zorder = 100, c=colors)
plt.show()

'''
new_centroidlist=list();
p0=0
p1=0
p2=0
for i in range(len(Y)):
    my_list=list()
    for j in range(len(var.centroids)):
        a=Y[i]-var.centroids[j]
        my_list.append(a)
    c1 = np.mean(my_list[0])
    d1 = np.mean(my_list[1])
    b1 = np.mean(my_list[2])
    if(abs(c1)<abs(d1) and abs(c1)<abs(b1)):
       p0 = p0 + 1
       if(var.labels[i]==0):
           ax.scatter(Y[i][0], Y[i][1], Y[i][2], c=colors[1])
    else:
        new_centroidlist.append(Y[i])
    

   if(abs(d1) < abs(c1) and abs(d1)<abs(b1) and var.labels[i]==1):
        p1 = p1 + 1
        if (var.labels[i] == 1):
            ax.scatter(Y[i][0], Y[i][1], Y[i][2], c=colors[1])
    else:
        new_centroidlist.append(Y[i])

    if (abs(b1) < abs(c1) and abs(b1) < abs(d1) and var.labels[i]==2):
        p2 = p2 + 1
        if (var.labels[i] == 2):
            ax.scatter(Y[i][0], Y[i][1], Y[i][2], c=colors[1])
    else:
        new_centroidlist.append(Y[i])
        

ax.scatter(var.centroids[:, 0],var.centroids[:, 1],var.centroids[:, 2], marker ="p", s=150, linewidths = 5, zorder = 100, c=colors)
plt.show()
'''
print("new samples")
print len(new_centroidlist)

print radi0
print radi1
print radi1
smallest_radi=0
if(radi0<radi1 and radi0<radi2):
    smallest_radi=radi0
elif(radi1<radi0 and radi1<radi2):
    smallest_radi=radi1

elif(radi2<radi0 and radi2<radi1):
    smallest_radi=radi2
print("smallest radi:")
print smallest_radi




if(flag==1):
    print("SAMPLES FALLING IN SAME CLUSTER 0")
    print(p0)
    print("outoff")
    print(len(Y))



if(flag==2):
    print("SAMPLES FALLING IN SAME CLUSTER 1")
    print(p1)
    print("outoff")
    print(len(Y))

p=(float)(p2*100)/len(Y)
if(flag==3):
    print("SAMPLES FALLING IN SAME CLUSTER 2")
    print(p2)
    print("Accuracy:")
    print("%f" % p)
    print('%')
    print("Total sample used for validation:")
    print(len(Y))


