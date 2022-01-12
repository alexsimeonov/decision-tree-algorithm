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
    
print("The Criterion used by the binning algorithm is Chi2")
max_depth = input("Enter desired max depth of decision tree: ")
min_samples_split = input("Enter minimum samples in a node after split: ")
min_samples_leaf = input("Enter minimum samples in a leaf node: ")
max_children_count = input("Enter maximum number of children per node: ")

hyperparams = {}

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

print("Root node ready. Growing starts.")
root_node.grow();
