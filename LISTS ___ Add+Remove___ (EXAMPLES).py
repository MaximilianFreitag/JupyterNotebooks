#!/usr/bin/env python
# coding: utf-8

# ### Remove 1 specific letter from list

# In[31]:


Liste = ['vbg', 'vkb', 'vv', 'vbj', 'vz'] #the list continues 


print([s.strip('v') for s in Liste]) # remove the 8 from the string borders
print([s.replace('v', '') for s in lst]) # remove all the 8s 


# ### Remove last string in Liste

# In[12]:


Liste = ['vbg', 'vkb', 'vv', 'vbj', 'vz']

NewListe = Liste[:-1]

print(NewListe)


# ### Add +1 or Characters for each element in Liste

# In[35]:


x = [1, 2, 3, 4, 5]

for i in range(len(x)):
     x[i] += 1

print(x)


# In[40]:


y = ['a','b','c']

for i in range(len(y)):
     y[i] += '######'

print(y)


# ### Unique values in 2D array

# In[3]:


import numpy as np
b = np.array([[1, 1, 0], [1, 1, 0], [2, 2, 4]])
np.unique(b, axis=1)


# ### Iterating over a list with string + integer

# In[8]:


d = [('red', 1), ('blue', 2), ('green', 3)]

for value in d: 
    word = value[0]
    value = value[1]
    print(word, value)
    
print('______________________')    
    
for value in d: 
    word = value[0]
    value = value[1]
    print("Word: {:}, Value: {:}".format(word , value))


# In[ ]:




