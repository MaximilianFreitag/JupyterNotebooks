#!/usr/bin/env python
# coding: utf-8

# ### Multiplication

# #### Python3.7

# In[3]:


from functools import reduce  # Required in Python 3.
from operator import mul

def prod(iterable): # This function multiplies all numbers of a row.
    return reduce(mul, iterable, 1)
    
#a = [(1,2,3), (2,3,4)]
#b = [5,6]
    
out = [prod(row) * elem for row, elem in zip(a, b)]

print(out)


# #### Python 3.8 +

# In[5]:


from math import prod
    
#a = [(1, 2, 3), (2, 3, 4)]
#b = [5, 6]
    
out = [prod(row) * elem for row, elem in zip(a, b)]

print(out)


# ### Devision - 1*3 mit einer 3*3 Matrix

# In[7]:


import numpy as np
import pandas as pd
    
a = np.array([[0.1562,0.0774,0.0702]])
b = np.array([
        [0.0365,0.0191,0.0217],
        [0.0191,0.0331,0.0292],
        [0.0217,0.0292,0.0591]])
        
        
x = np.linalg.lstsq(b.T,a.T)
print(x)


(array([[ 4.49111376],
       [ 0.2724206 ],
       [-0.59580119]]), array([], dtype=float64), 3, array([0.09268238, 0.02342602, 0.0125916 ]))


# In[ ]:




