'''Q5 Decision Tree'''
# import standard packages and self-built class and variables
import pandas as pd

from coursework1 import data, all_sourceIP, all_destIP, all_classes
from coursework1 import S1, S2, S3, S4, D1, D2, D3, D4
from tree import DecisionTreeClassifier

# data pre-processing
# replacing all IP addresses with corresponding symbols
DT_data = [] # dataset used in decision tree
for index in range(len(data)):
    if all_sourceIP[index] in S1:
        source_cluster = 's1' # s1 represents cluster 1 in source IP
    elif all_sourceIP[index] in S2:
        source_cluster = 's2'  # s2 represents cluster 2 in source IP
    elif all_sourceIP[index] in S3:
        source_cluster = 's3'  # s3 represents cluster 3 in source IP
    elif all_sourceIP[index] in S4:
        source_cluster = 's4'  # s4 represents cluster 4 in source IP

    if all_destIP[index] in D1:
        dest_cluster = 'd1'  # d1 represents cluster 1 in dest IP
    elif all_destIP[index] in D2:
        dest_cluster = 'd2' # d2 represents cluster 2 in dest IP
    elif all_destIP[index] in D3:
        dest_cluster = 'd3' # d3 represents cluster 3 in dest IP
    elif all_destIP[index] in D4:
        dest_cluster = 'd4' # d4 represents cluster 4 in dest IP

    DT_data.append([source_cluster, dest_cluster, all_classes[index]]) # one example consists of 3 elements, the last column is label
DT_data = pd.DataFrame(DT_data, columns=['sourceIP', 'destIP', 'classification']) # package original data as pd.DataFrame
DT_data = DT_data.sample(frac=1).reset_index(drop=True)  # shuffle data
training_data = DT_data.iloc[0:int(len(DT_data) * 0.7), :] # split training data and test data(0.7:0.3)
test_data = DT_data.iloc[int(len(DT_data) * 0.7 ): len(DT_data), :].reset_index(drop=True) # test data
my_tree = DecisionTreeClassifier() # instance of class DecisionTreeClassifier
tree_dict = my_tree.create_tree(DT_data, DT_data.columns[0:-1]) # use the method create_tree build a decision tree
print(tree_dict) # print the tree as dict
dot_file = my_tree.dotify() # convert the dict to dot file
with open('./results/my_tree1.dot', 'w') as f: # write dot file to local
    dot = my_tree.dotify()
    f.write(dot)
my_tree.visualize('./results/my_tree1.dot') # use method visualize to plot decision tree
