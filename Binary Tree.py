#!/usr/bin/env python
# coding: utf-8

# ### Inverting Binary Tree (recursive)

# > **pip install binarytree (In your terminal)**

# In[23]:


#Creating the binary tree
from binarytree import build
from binarytree import tree
   
# List of nodes 
nodes =[4, 2, 7, 1, 3, 6, 9] 
  
# Builidng the binary tree 
binary_tree = build(nodes) 
print('Binary tree from example :\n ', 
      binary_tree) 
  
# Getting list of nodes from 
# binarytree 
print('\nList from binary tree :',  
      binary_tree.values) 


# #### Pseudo code

# In[ ]:


#def invert_tree(nodes, root)

#Stopping recursion if tree is empty

#swap left subtree with right subtree

#invert left subtree

#invert right subtree


# In[45]:



class Node: 
  
    # constructor to create a new node   
    def __init__(self, nodes): 
        self.nodes = data 
        self.left = None
        self.right = None
  
# A utility function swap left node and right node of tree 
# of every k'th level  
def swapEveryKLevelUtil(root, level, k): 
      
    # Base Case  
    if (root is None or (root.left is None and
                        root.right is None ) ): 
        return 
  
    # If current level+1 is present in swap vector 
    # then we swap left and right node 
    if (level+1)%k == 0: 
        root.left, root.right = root.right, root.left 
      
    # Recur for left and right subtree 
    swapEveryKLevelUtil(root.left, level+1, k) 
    swapEveryKLevelUtil(root.right, level+1, k) 
  
      
# This function mainly calls recursive function 
# swapEveryKLevelUtil 
def swapEveryKLevel(root, k): 
      
    # Call swapEveryKLevelUtil function with  
    # initial level as 1 
    swapEveryKLevelUtil(root, 1, k) 
  
# Method to find the inorder tree travesal 
def inorder(root): 
      
    # Base Case 
    if root is None: 
        return 
    inorder(root.left) 

print(swapEveryKLevelUtil(0, 0, nodes))


# In[ ]:




