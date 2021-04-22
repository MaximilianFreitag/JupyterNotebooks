#!/usr/bin/env python
# coding: utf-8

# ### Big O notation

# The big O notation is used to denote the asymptotic upper bound of an algorithm as the input size (n) tends to infinity. 
# 
# In this particular case, there are only a finite number of possible solutions (45) which your algorithm correctly finds. Therefore the worst-case running time (or memory consumption) does not depend on the inputs; you always need to loop over maximum 45 candidates.

# In[13]:


Liste = [1,2,3,4,5,6]


# #### Constant Time
# 

# In[14]:


print(Liste[0])


# #### Linear Time

# In[15]:


for i in Liste:
    if i == 6:
        print(i)


# #### Quardatic time

# In[16]:


for i in Liste:
    for j in Liste:
        i += 1
        j += 4
        print(i,j)


# #### Logarithmic Time

# In[ ]:





# ### Runtime

# In[ ]:





# In[ ]:




