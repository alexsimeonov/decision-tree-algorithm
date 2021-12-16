# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:38:36 2021

@author: alexs
"""

class Node:
    def __init__(self, data_set: list):
        #the results from ub
        self.data_set = data_set
        self.left = None
        self.right = None
    
    def split(self, split_method):
            #call the external function (split_method) from main.py 
            #with self.data_set use the result in order to create a new Node instances
            #for left and right properties and call their split methods
            #this should continue until one of the predefined conditions is reached
            
        