# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 00:23:35 2021

@author: alexs
"""

# From node import Node
# Import initial dataset

# Accepting hyperparametes input from user -> 
# only the ones with values would be taken under consideration
criterion = input("Enter the name of function to measure split quality: ") #should be limited to gini or entropy, maybe default should be gini
max_depth = input("Enter desired max depth of decision tree: ")
min_samples_split = input("Enter minimum samples in a node after split: ")
min_samples_leaf = input("Enter minimum samples in a leaf node: ")

# Here the imported dataset should be passed to the root node, along with the rest of the desired
# hyperparameters. The result should be stored in a variable in order to be used/visualised at the end.
# Rest of the logic is described in node.py