from math import log2
import pandas as pd
import uuid
import pickle
from collections import namedtuple
import graphviz

class DecisionTreeClassifier():


    def split_dataset(self, dataset, feature_index): # define method to split subset as feature
        dataset = pd.DataFrame(dataset) # if the type of dataset is not DataFrame, transfer it to DataFrame
        subset1, subset2, subset3, subset4= [],  [],  [],  [] # define 4 cluster to store the index of subset
        instances = dataset.iloc[:, feature_index] # get the examples of  each feature
        split_dict = {} # define the dict to store feature value and its corresponding subset

        # feature_value = sorted(set(instances))
        # if feature_index == 0:
        #     feature_value = ['s1', 's2', 's3', 's4']
        # else:
        #     feature_value = ['d1', 'd2', 'd3', 'd4']

        features = list(dataset.columns)[0:]
        feature_value = sorted(list(set(dataset[features[feature_index]]))) # return the feature values(symbols)

        for index in list(dataset.index):
            if instances[index] == feature_value[0]: # split the dataset according to feature value
                subset1.append(index)
            elif instances[index] == feature_value[1]:
                subset2.append(index)
            elif instances[index] == feature_value[2]:
                subset3.append(index)
            elif instances[index] == feature_value[3]:
                subset4.append(index)
        subset = [subset1, subset2, subset3, subset4]
        for i in range(len(feature_value)):
            split_dict[feature_value[i]] = subset[i] # key = feature symbol, value = subset of each feature value

        # split_dict = {'subset1':subset1, 'subset2':subset2, 'subset3':subset3, 'subset4':subset4}

        return split_dict, subset # return split subsets


    def get_entropy(self, classes): # compute the entropy in a dataset
        counts = pd.value_counts(classes) # return the number of each classification
        probs = [v/len(classes) for v in counts] # compute the probability of each classification
        entropy = sum(-prob * log2(prob) for prob in probs) # compute the entropy according to the formula

        return entropy


    def choose_feature(self, dataset): # use the whole dataset: DT_data, class = DT_data['classification']
        '''compute the entropy gain to choose the best split feature'''
        dataset = pd.DataFrame(dataset) # transfer the type of dataset to DataFrame
        base_entropy = self.get_entropy(dataset['classification']) # get the base entropy of original dataset
        feature_num = dataset.shape[1] - 1 # return the number of features, since the last column is the label
        entropy_gains = []
        for feature in range(feature_num):
            new_entropy = 0
            _, subset = self.split_dataset(dataset, feature) # split the dataset according to feature
            if len(dataset) != 0:
                for i in range(len(subset)):
                    weight = len(subset[i])/len(dataset) # compute the weight of each subset
                    # compute the entropy of each subset and the total entropy of all subsets splited by feature
                    new_entropy += (weight * self.get_entropy(dataset.loc[subset[i]]['classification']))
                entropy_gains.append(base_entropy - new_entropy) # compute the entropy gain
            else:
                entropy_gains.append(0) # when one of the subsets is empty

        return entropy_gains.index(max(entropy_gains))  # return the index of max entropy, O:'sourceIP', 1:'destIP'


    def get_majority_class(self, dataset): # return the major class in a dataset
        classes = dataset['classification']
        counts = pd.value_counts(classes).to_dict() # convert pd.Series to dict {class:counts}

        return max(counts)



    def create_tree(self, dataset, feature_names): # use the whole dataset
        '''use the recursive method to create the decision tree '''
        dataset = pd.DataFrame(dataset)  # transfer the type of dataset to DataFrame
        classes = dataset['classification'] # get the classifications of dataset
        # feature_names = list(dataset.columns)[0:-1]
        feature_names = list(feature_names)

        # stop conditions
        if len(set(classes)) == 1: # if the dataset has only one classification
            return set(classes) # return the name of classification
        if len(dataset) == 0: # if the dataset is empty
            return None # return none
        if len(feature_names) == 0: # if we have splited all features
            if len(set(classes)) != 1: # but there still are more than 1 classification
                probs= []
                for i in range(len(set(classes))):
                    probs.append([list(set(classes))[i],round(pd.value_counts(classes)[i] / len(classes), 4)]) # return the probability of each class
                # return self.get_majority_class(dataset)
                # return list(set(classes))
                return probs


        # create tree, return in dict
        tree = {}
        best_feature_index = self.choose_feature(dataset) # choose the best feature to split
        feature = feature_names[best_feature_index]
        tree[feature] = {} # create the sub dict to store sub tree
        # create the subset of data
        sub_feature_names = feature_names[:]
        sub_feature_names.pop(best_feature_index) # once the feature has been chosen, it will be pop out

        split_dict, subset = self.split_dataset(dataset, best_feature_index) # split dataset according to feature
        for feature_value, sub_dataset in split_dict.items(): # feature_value = 's1'...'s4' or 'd1'...'d4'
            # if feature_value == list(split_dict.keys())[-1]:
            #     # feature_names.pop(best_feature_index)
            #     dataset = dataset.drop([feature], axis=1)

            if len(dataset) != 0: # if dataset is not empty
                tree[feature][feature_value] = self.create_tree(dataset.loc[sub_dataset], sub_feature_names) # recursively run create_tree to create decision tree
            else:
                continue

        self.tree = tree
        self.feature_names = feature_names

        return tree


    def get_nodes_edges(self, tree=None, root_node=None):
        '''return nodes and edges of tree'''
        Node = namedtuple('Node', ['id', 'label']) # define a namedtuple to store nodes
        Edge = namedtuple('Edge', ['start', 'end', 'label']) # define a namedtuple to store edges

        if tree is None:
            tree = self.tree

        if type(tree) is not dict:
            return [], []

        nodes, edges = [], []

        if root_node is None: # if parameter root_node is empty
            label = list(tree.keys())[0] # always return the first key of tree dict
            root_node = Node._make([uuid.uuid4(), label]) # use the function uuid4() generate a unique ID
            nodes.append(root_node)

        for edge_label, sub_tree in tree[root_node.label].items(): # return the key and value of sub tree
            if sub_tree == None:
                continue
            else:
                node_label = list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree # set the feature value as the node label
                sub_node = Node._make([uuid.uuid4(), node_label]) # generate the sub node
                nodes.append(sub_node)

                edge = Edge._make([root_node, sub_node, edge_label]) # assign the value to Edge
                edges.append(edge)

                sub_nodes, sub_edges = self.get_nodes_edges(sub_tree, root_node=sub_node) # recursively run the get nodes and edges untill end
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)

        return nodes, edges



    def dotify(self, tree=None): # convert the dict to dot content
        if tree is None:
            tree = self.tree

        content = 'dot file decision_tree {\n'
        nodes, edges = self.get_nodes_edges(tree) # return the nodes and edges from get_nodes_edges function

        for node in nodes:
            content += '    "{}" [label="{}"];\n'.format(node.id, node.label) # write down the node id and node label

        for edge in edges:
            start, label, end = edge.start, edge.label, edge.end
            content += '    "{}" -> "{}" [label="{}"];\n'.format(start.id, end.id, label) # write down the start node id, end node id and the end label
        content += '}'


        return content

    def visualize(self, file_path):
        '''visualize the tree using dot file and graphviz'''
        with open(file_path) as f: # read the dot file as Chars
            dot_graph = f.read()

        dot = graphviz.Source(dot_graph) # use built-in function Source
        dot.view() # view tree as pdf

    def tree_classify(self, dataset, feature_names=None, tree=None): # full dataset, the last column is label
        '''use the tree to classify data'''

        if tree is None: # read the tree dict
            tree = self.tree

        if feature_names is None: # read the feature name
            feature_names = self.feature_names

        if type(tree) is not dict:
            return tree

        # if isinstance(dataset, pd.DataFrame): # if the input more than one example,
        #     results = []
        #     dataset = pd.DataFrame(dataset)
        #     for index in list(dataset.index):
        #         data = list(dataset.loc[index,:])
        #         feature = list(tree.keys())[0]
        #         value = data[feature_names.index(feature)]
        #         sub_tree = tree[feature][value]
        #         results.append(self.tree_classify(dataset, feature_names, sub_tree))
        #     return results
        #
        # else: # only one row of data
        dataset =list(dataset) # read the input data as list
        feature = list(tree.keys())[0] # get the feature name from tree
        value = dataset[feature_names.index(feature)] # get the feature value
        sub_tree = tree[feature][value]# step into the sub dict/tree
        results = self.tree_classify(dataset, feature_names, sub_tree) # recursively run

        return results


    def dump_tree(self, filename, tree=None):
        '''store the tree to local'''

        if tree is None:
            tree = self.tree

        with open(filename, 'wb') as f:
            pickle.dump(tree, f)

    def load_tree(self, filename):
        '''load tree from local'''

        with open(filename, 'rb') as f:
            tree = pickle.load(f)
            self.tree = tree
        return tree
