# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 00:23:35 2021

@author: alexs
"""

from node import Node
from enum import Enum
import pandas as pd

csv_file = 'C:/University/Дипломна работа/CODE/decision-tree-algorithm/binning/aislab/data/cnf_LC.csv'

class Criterion(Enum):
    GINI = "gini"
    ENTROPY = "entropy"
    
# Accepting hyperparametes input from user ->
# only the ones with values would be taken under consideration
# Remove criterion input
criterion = input("Enter the name of function to measure split quality(Could be 'gini' or 'entropy'): ") #should be limited to gini or entropy, maybe default should be gini
max_depth = input("Enter desired max depth of decision tree: ")
min_samples_split = input("Enter minimum samples in a node after split: ")
min_samples_leaf = input("Enter minimum samples in a leaf node: ")
max_children_count = input("Enter maximum number of children per node: ")
# add max number of child nodes

# Here the imported dataset should be passed to the root node, along with the rest of the desired
# hyperparameters. The result should be stored in a variable in order to be used/visualised at the end.
# Rest of the logic is described in node.py

hyperparams = {}

# Used to check if user input for criterion is valid Criterion
values = set(item.value for item in Criterion)

if criterion:
    # instead of default value use error handling
    # Please send me a list of the criterions supported by the algorithm in order to add them and error handling
    hyperparams["criterion"] = Criterion(criterion) if (criterion in values) else Criterion.GINI
if max_depth:
    hyperparams["max_depth"] = max_depth
if min_samples_split:
    hyperparams["min_samples_split"] = min_samples_split
if min_samples_leaf:
    hyperparams["min_samples_leaf"] = min_samples_leaf
if max_children_count:
    hyperparams["max_children_count"] = max_children_count
    
initial_dataset = pd.read_csv(csv_file)
    
# Creating the root node from data provided along with user input.
root_node = Node(hyperparams, initial_dataset)
