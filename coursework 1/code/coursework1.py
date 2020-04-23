import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv('coursework1.csv')

## Q1: Basic Data Processing
all_sourceIP = data['sourceIP'].tolist() # convert all source IP to list
all_destIP = data['destIP'].tolist() # convert all source IP to list
all_classes = data['classification'].tolist() # convert all classifications to list
distinct_S_IP = set(all_sourceIP) # convert list to set, because set has no duplicate elements
distinct_D_IP = set(all_destIP) # convert list to set, because set has no duplicate elements
distinct_classes = set(all_classes)  # convert list to set, because set has no duplicate elements
print(len(all_sourceIP)) # print the original number of source IP
print(len(all_destIP)) # print the original number of destination IP
print(len(all_classes)) # print the original number of classifications
print('the number of distinct source IP addresses: ' + str(len(distinct_S_IP))) # print the distinct source IP addresses
print('the number of distinct destination IP addresses: ' + str(len(distinct_D_IP))) # print the distinct destination IP addresses
print('the number of distinct classifications: ' + str(len(distinct_classes))) # print the distinct classifications IP

## Q2: Basic Data Analysis and Visualisation
from pyecharts import options as opts
from pyecharts.charts import Bar # import the pyecharts to draw Bar chart
from typing import List

def count_values(nums: List[int]):
    ''' count how many times of each element appears in the list '''
    if len(nums) == 0:
        return 0
    else:
        count = []
        for i in set(nums):
            count.append(nums.count(i))
        return count

B1 = (
    Bar() # initialize parameters
    .add_xaxis(xaxis_data = distinct_S_IP) # set x axis data
    .add_yaxis(
        series_name='Source IP Addresses', # set the name of legend
        yaxis_data= count_values(all_sourceIP), # set y axis data
        is_selected=True,  # whether choose the legend
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title='Group Source IP Addresses'), # set the title of chart
        datazoom_opts=opts.DataZoomOpts(type_='inside'), # activate the zoom function
    )
    .render('results/SourceIP_histogram.html') # render the bar chart into html file
)
B2 = (
    Bar() # initialize parameters
        .add_xaxis(xaxis_data=distinct_D_IP) # set x axis data
        .add_yaxis(
        series_name='Destination IP Addresses', # set the name of legend
        yaxis_data=count_values(all_destIP), # set y axis data
        is_selected=True, # whether choose the legend
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title='Group Destination IP Addresses'), # set the title of chart
        datazoom_opts=opts.DataZoomOpts(type_='inside'), # activate the zoom function
    )
        .render('results/DestinationIP_histogram.html') #  render the bar chart into html file
)
B3 = (
    Bar() #initialize parameters
        .add_xaxis(xaxis_data=distinct_classes) # set x axis data
        .add_yaxis(
        series_name='Classifications', # set the name of legend
        yaxis_data=count_values(all_classes), # set y axis data
        is_selected=True, # whether choose the legend
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title='Group Classifications'), # set the title of chart
        datazoom_opts=opts.DataZoomOpts(type_='inside'), # activate the zoom function
    )
        .render('results/Classification_histogram.html')  # render the bar chart into html file
)


## Q3: Clustering
# K-means
from clustering import MyKMeans #import my own code on clustering

distinct_S_IP = list(distinct_S_IP) # convert the type of distinct_S_IP into list
kmeans_sourceIP = MyKMeans() # create an instance of MyKeans on sourceIP
sourceIP_count = kmeans_sourceIP.count_values(all_sourceIP) # use the method of MyKmeans to calculate the number of each IP address
X_sourceIP = kmeans_sourceIP.merge_data(distinct_S_IP, sourceIP_count) # combine the each IP address with its number as the original data of kmeans
kmeans_sourceIP.draw_elbow(X_sourceIP, 10, 'SourceIP_elbow') # use the elbow to select best number of cluster
kmeans_sourceIP.plot_cluster(X_sourceIP, 2, 'sourceIP') # k=2 is the best cluster for source IP, plot the clusters

distinct_D_IP = list(distinct_D_IP) # convert the type of distinct_D_IP into list
kmeans_destIP = MyKMeans() # create an instance of MyKeans on destIP
destIP_count = kmeans_destIP.count_values(all_destIP) # use the method of MyKmeans to calculate the number of each IP address
X_destIP = kmeans_destIP.merge_data(distinct_D_IP, destIP_count) # combine the each IP address with its number as the original data of kmeans
kmeans_destIP.draw_elbow(X_destIP, 10, 'destIP_elbow') # use the elbow to select best number of cluster
kmeans_destIP.plot_cluster(X_destIP, 2, 'destIP') # k=2 is the best cluster for source IP, plot the clusters


# Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage # import the hierarchy package

linked1 = linkage(X_sourceIP, 'average') # compute the average distance between two clusters
labelList = range(0, len(X_sourceIP)) # set labels
plt.figure(figsize=(10, 7)) # change the size of canvas
dendrogram(linked1, labels=labelList) # draw the dendrogram
plt.xlabel('X') # set x label
plt.ylabel('Number of each source IP') # set y label
plt.title('The dendogram for source IP') # set title of figure
plt.savefig('results/sourceIP_dendogram.png') # save the image to local
plt.show() # show plot

linked2 = linkage(X_destIP, 'average')  # compute the average distance between two clusters
labelList = range(0, len(X_destIP)) # set labels
plt.figure(figsize=(10, 7))# change the size of canvas
dendrogram(linked2, labels=labelList)# draw the dendrogram
plt.xlabel('X')# set x label
plt.ylabel('Number of each destination IP')# set y label
plt.title('The dendogram for destination IP')# set title of figure
plt.savefig('results/destIP_dendogram.png')# save the image to local
plt.show()# show plot


## Q4 Finding Relationship
from findingrelation import FindRelation # import my own class

def data_to_dic(input, count): # convert the type of data into dict key=ip value = counts
    X = {}
    for index in range(len(input)):
        X[input[index]] = count[index]
    return X

sourceIP = data_to_dic(distinct_S_IP, sourceIP_count) # convert source IP and its counts to dict type
sourceIP = sorted(sourceIP.items(), key=lambda item:item[1]) # sort the dict according to its values
sourceIP = dict(sourceIP)
S1,S2,S3,S4 = [],[],[],[] # create 4 clusters to store IP addresses

for key, value in sourceIP.items(): # Clustering source IP addresses according to their records
    if value <= 20: # cluster1 less than 20
        S1.append(key)
    elif value >=21 and value <= 200: # cluster 2 between 20 and 200
        S2.append(key)
    elif value >= 201 and value <= 400: # cluster 3 between 200 and 400
        S3.append(key)
    elif value > 400: # cluster 4 large than 400
        S4.append(key)

destIP = data_to_dic(distinct_D_IP, destIP_count) # convert destination IP and its counts to dict type
destIP = sorted(destIP.items(), key=lambda item:item[1]) # sort the dict according to its values
destIP = dict(destIP)
D1,D2,D3,D4 = [],[],[],[]  # create 4 clusters to store IP addresses

for key, value in destIP.items(): # Clustering destination IP addresses according to their records
    if value <= 40: # cluster1 less than 40
        D1.append(key)
    elif value >=41 and value <= 100: # cluster 2 between 40 and 100
        D2.append(key)
    elif value >= 101 and value <= 400: # cluster 3 between 100 and 400
        D3.append(key)
    elif value > 400: # cluster 4 large than 400
        D4.append(key)

'''      destIP
            D1     D2      D3      D4     
sourceIP
    S1      p11   p12     p13     p14
    
    S2      p21   p22     p23     p24
    
    S3      p31   p32     p33     p34
    
    S4      p41   p42    p43      p44

conditional probabilities: p(destIP|sourceIP)
'''

sourceIP_cluster = [S1, S2, S3, S4] # set the 4 clusters together as a list
destIP_cluster = [D1, D2, D3, D4] # set the 4 clusters together as a list
source_data = data.loc[:, ['sourceIP', 'destIP']]  # get the source data which consists of two columns from original dataset
Q4 = FindRelation() # instance the class FindRelation()
prob_table1 = np.zeros((4,4)) # set a new matrix to store the conditinal probability of source IP given dest IP p(sourceIP | destIP)
for i in range(0, 4):
    for j in range(0, 4): # cluster index of condition
        # compute the conditional probability through calculate_probabilities
        prob_table1[i,j] = Q4.calculate_probabilities(source_data['destIP'], destIP_cluster[j], source_data['sourceIP'], sourceIP_cluster[i])
print('This is the conditional probability of sourceIP given destIP:\n', prob_table1)

prob_table2 =np.zeros((4,4)) # set a new matrix to store the conditinal probability of dest IP given source IP p(destIP | sourceIP)
for i in range(0, 4): # cluster index of condition
    for j in range(0, 4):
        # compute the conditional probability through calculate_probabilities
        prob_table2[i,j] = Q4.calculate_probabilities(source_data['sourceIP'], sourceIP_cluster[i], source_data['destIP'], destIP_cluster[j])
print('\nThis is the conditional probability of destIP given sourceIP:\n', prob_table2)





