from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import List


class MyKMeans():

    def count_values(self, nums: List[int]):
        ''' count the number of distinct value in a list'''
        if len(nums) == 0:
            return 0
        else:
            count = []
            for i in set(nums):
                count.append(nums.count(i))
            return count

    def merge_data(self, input, count):
        ''' because the IP addresses are to long, replace them with integers'''
        if len(input) == 0:
            return 0
        else:
            X = []
            for index in range(len(input)):
                X.append([index + 1, count[index]])
            X = np.array(X, dtype=np.int) # transfer to np.array
            return X



    def draw_elbow(self, X, K, filename):
        ''' use the elbow plot to choose the best number of cluster'''
        distortion = []
        K_range = range(1, K)
        for k in K_range:
            kmeanModel = KMeans(n_clusters=k).fit(X) # set a kmeans model use scipy
            kmeanModel.fit(X) # run kmeans
            # compute the distance between data point and cluster centroids
            distortion.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean')**2, axis=1)) / X.shape[0])

        # plot the elbow
        plt.figure()
        plt.plot(K_range, distortion, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method for '+str(list(filename.split('_'))[0])  +' showing the optimal k')
        plt.savefig('results/' + filename+'.png')
        plt.show()


    def plot_cluster(self, X, k, title):
        ''' plot clusters'''
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)  # run clustering
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
        plt.title('Clusters for ' + title)
        plt.show()