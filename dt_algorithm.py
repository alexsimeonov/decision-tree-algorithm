# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 00:23:35 2021

@author: alexs
"""
import os
import sys

from node import Node
from decision_tree import DecisionTree
from enum import Enum

wpath = '/Users/alexandersimeonov/Library/Python/2.7/lib/python/site-packages'
# path to dataset
x_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/binning/aislab/data/x_samples_250000.csv'
# path to output values
y_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/binning/aislab/data/y_samples_250000.csv'
c_path = '/Users/alexandersimeonov/Documents/Development/University/decision-tree-algorithm/binning/aislab/data/cnf_LC.csv'

sys.path.append(wpath)
os.chdir(wpath)

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

decision_tree = DecisionTree(x_path, y_path, c_path, hyperparams,)
decision_tree.grow()
