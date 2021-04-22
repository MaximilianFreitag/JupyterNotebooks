#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random

a = []

for x in range(1000):

    a.append(random.randint(1,501))
    
print(' '.join(map(str, a)))

print(a)


# In[ ]:




