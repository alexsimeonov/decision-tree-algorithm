# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:38:36 2021

@author: alexs
"""
from enum import Enum
import uuid

# Currently having problems with following lines - ask for some help -> why it says that gnrl is missing
#from binning.aislab.dp_feng.binenc import *
#from binning.aislab.gnrl import *

# The node status -> could be improved and make sure it works
class Status(Enum):
    TERMINAL = 0
    LEAF = 1

class Node:
    def __init__(self, data_set: list, params: list, parent=None):
        # The results from ub
        self.id = uuid.uuid1() # Generates a UUID based on the host ID and current time
        self.data_set = data_set
        self.params = params
        self.parent = parent # If present
        # Ask if the child nodes should be limited to 2 and if so -> is the following design appropriate
        # Else use a list 'children' maybe
        self.left = None
        self.right = None
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
        