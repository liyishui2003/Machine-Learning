from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

def Kmeans(x1,n):
    kmeans = KMeans(n_clusters=n) #n个聚类中心
    kmeans.fit(x1)
    y_kmeans = kmeans.predict(x1)
    centers = kmeans.cluster_centers_    #聚类中心
    return y_kmeans,centers

fig = plt.figure()

#make_circles聚类
#聚类前
x1, y1 = make_circles(n_samples=1000, factor=0.2, noise=0.1)  
plt.title('make_circles')
plt.scatter(x1[:,0],x1[:,1],marker='*',c=y1,cmap='seismic') 
plt.title("make_circles_102102103liyishui")
plt.show()
#聚类后，下同
y_kmeans,centers = Kmeans(x1,2)
plt.scatter(x1[:,0],x1[:,1],marker='*',c=y_kmeans,cmap='seismic')
plt.scatter(centers[:,0], centers[:,1],marker='*',c='black',s=80)
plt.show()

acc=accuracy_score(y1,y_kmeans)               #实际标签和通过算法获得的预测标签之间的精度
nmi=normalized_mutual_info_score(y1,y_kmeans) #标准化互信息
ari=adjusted_rand_score(y1,y_kmeans)          #调整的兰德系数
print("ACC = ",acc)
print("NMI = ",nmi)
print("ARI = ",ari)


#make_moons聚类
x1,y1 = make_moons(n_samples=1000,noise=0.1)   
plt.title('make_moons_102102103liyishui') 
plt.scatter(x1[:,0],x1[:,1],marker='*',c=y1,cmap='seismic')  
plt.show()

y_kmeans,centers = Kmeans(x1,2)
plt.scatter(x1[:,0],x1[:,1],marker='*',c=y_kmeans,cmap='seismic')  
plt.scatter(centers[:,0], centers[:, 1],marker='*',c='black',s=80)
plt.title("make_moons_102102103liyishui")
plt.show()

acc=accuracy_score(y1,y_kmeans)
nmi=normalized_mutual_info_score(y1,y_kmeans)
ari=adjusted_rand_score(y1,y_kmeans)
print("ACC = ",acc)
print("NMI = ",nmi)
print("ARI = ",ari)


#make_blobs聚类
x1,y1 = make_blobs(n_samples=4000,n_features=2,centers=2)
plt.title('make_blobs_102102103liyishui')
plt.scatter(x1[:,0],x1[:,1],marker='*',c=y1,cmap='seismic')
plt.show()

y_kmeans,centers = Kmeans(x1,2)
plt.scatter(x1[:,0],x1[:,1],marker='*',c=y_kmeans,cmap='seismic')
plt.scatter(centers[:,0], centers[:, 1],marker='*',c='black',s=80)
plt.title("make_blobs_102102103liyishui")
plt.show()

acc=accuracy_score(y1,y_kmeans)
nmi=normalized_mutual_info_score(y1,y_kmeans)
ari=adjusted_rand_score(y1,y_kmeans)
print("ACC = ",acc)
print("NMI = ",nmi)
print("ARI = ",ari)





#====================problem2=================


#stones.jpg

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import PIL.Image as image
#读取原始图像
paths ="stones.jpg"
X = plt.imread(paths)
X = np.array(X)
#print(X.shape)
shape = row ,col ,dim =X.shape
X_ = X.reshape(-1,3)#(将矩阵化为2维，才能使用聚类)
#print(X_.shape)
def kmeans(X, n):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    Y = kmeans.predict(X)
    return Y

plt.figure(1)  # 图像窗口名称
plt.subplot(2,3,1)
plt.imshow(X)
plt.axis('off')  # 关掉坐标轴为 off
plt.xticks([])
plt.yticks([])
plt.title("Picture_102102103liyishui")
for t in range(2, 7):
    index = '23' + str(t)
    plt.subplot(int(index))
    label = kmeans(X_,t)
    print("label.shape=",label.shape)
    # get the label of each pixel
    label = label.reshape(row,col)
    # create a new image to save the result of K-Means
    pic_new = image.new("RGB", (col, row))#定义的是图像大小为y*x*3的图像，这里列在前面行在后面
    for i in range(col):
        for j in range(row):
                if label[j][i] == 0:
                    pic_new.putpixel((i, j), (0, 0, 255))#填写的是位置为（j,i）位置的像素，列和行也是反的
                elif label[j][i] == 1:
                    pic_new.putpixel((i, j), (255, 0, 0))
                elif label[j][i] == 2:
                    pic_new.putpixel((i, j), (0, 255, 0))
                elif label[j][i] == 3:
                    pic_new.putpixel((i, j), (60, 0, 220))
                elif label[j][i] == 4:
                    pic_new.putpixel((i, j), (249, 219, 87))
                elif label[j][i] == 5:
                    pic_new.putpixel((i, j), (167, 255, 167))
                elif label[j][i] == 6:
                    pic_new.putpixel((i, j), (216, 109, 216))
    title = "k="+str(t)
    plt.title(title)
    plt.imshow(pic_new)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.xticks([])
    plt.yticks([])

plt.show()



#==================problem3=================


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
import os
import pandas as pd
from scipy.optimize import linear_sum_assignment
#from sklearn.utils.linear_assignment_ import linear_assignment

# -*- coding: utf-8 -*-
"""
Created on April 7, 2020

@author: Shiping Wang
  Email: shipingwangphd@gmail.com
  Date: April 14, 2020.
"""

from sklearn import metrics
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
'''
   Clustering accuracy
'''
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[ind[0][i], ind[1][i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size

'''
 Evaluation metrics of clustering performance
      ACC: clustering accuracy
      NMI: normalized mutual information
      ARI: adjusted rand index
'''


def getinfo():
    # 获取文件并构成向量
    #预测值为1维，把一张图片的三维压成1维，那么n张图片就是二维
    global total_photo
    file = os.listdir(r'D:\face_images')
    i = 0
    for subfile in file:
        photo = os.listdir(r'D:\face_images\\' + subfile)  #文件路径自己改
        for name in photo:
            photo_name.append(r'D:\face_images\\'+ subfile+'\\'+name)
            target.append(i)
        i += 1
    for path in photo_name:
        photo = imgplt.imread(path)
        photo = photo.reshape(1, -1)
        photo = pd.DataFrame(photo)
        total_photo = total_photo.append(photo, ignore_index=True)
    total_photo = total_photo.values
def kmeans():
    clf = KMeans(n_clusters=10)
    clf.fit(total_photo)
    y_predict = clf.predict(total_photo)
    centers = clf.cluster_centers_
    result = centers[y_predict]
    result = result.astype("int64")
    result = result.reshape(200, 200, 180, 3)#图像的矩阵大小为200,180,3
    return result,y_predict

def draw():
    fig,ax  = plt.subplots(nrows=10,ncols=20,sharex = True,sharey = True,figsize = [15,8],dpi = 80)
    plt.subplots_adjust(wspace = 0,hspace = 0)
    count = 0
    for i in range(10):
        for j in range(20):
            ax[i,j].imshow(result[count])
            count += 1

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('made_by_102102103liyishui')
    plt.show()

def clusteringMetrics():
    # Clustering accuracy
    ACC = cluster_acc(target, y_predict)

    # Normalized mutual information
    NMI = metrics.normalized_mutual_info_score(target, y_predict)

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(target, y_predict)
    
    print(" ACC = ", ACC)
    print(" NMI = ", NMI)
    print(" ARI = ", ARI)
    return ACC, NMI, ARI

    
    
if __name__ == '__main__':
    photo_name = []
    target = []
    total_photo = pd.DataFrame()
    getinfo()
    result,y_predict = kmeans()
    clusteringMetrics()
    draw()

