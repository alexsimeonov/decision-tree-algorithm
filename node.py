# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:38:36 2021

@author: alexs
"""
from enum import Enum
import uuid

#from aislab.dp_feng.binenc import *
#from aislab.gnrl import *
from binning.aislab import *

class Status(Enum):
    DECISION = 0
    TERMINAL = 1
    LEAF = 2

class Node:
    def __init__(self, params: list, data_set=None, parent=None):
        self.id = uuid.uuid1() # Generates a UUID based on the host ID and current time
        self.data_set = data_set
        self.params = params
        self.parent = parent # If present
        self.children = None # Add children after split
        self.status = Status.TERMINAL
    
    def grow(self, growing_method):
        print("growing")
            # When fix the import use the ubng method in here
        
            # Recursively executed the actions function described below
            # Call the external function (split_method) from main.py 
            # with self.data_set use the result in order to create a new Node instances
            # for left and right properties and call their split methods
            # This should continue until one of the predefined conditions is reached
            
    def prune():
        print("pruning")
        #for the pruning phase, not yet implemented
        