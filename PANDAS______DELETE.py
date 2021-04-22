#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

data = { 
    'A':['A1', 'A2', 'A3', 'A4', 'A5'],  
    'B':['B1', 'B2', 'B3', 'B4', 'B5'],  
    'C':['C1', 'C2', 'C3', 'C4', 'C5'],  
    'D':['D1', 'D2', 'D3', 'D4', 'D5'],  
    'E':['E1', 'E2', 'E3', 'E4', 'E5'] } 

df4 = pd.DataFrame(data) 

df4


# In[4]:


df4.drop(['A'], axis = 1) 


# ### Replace a value within table

# In[5]:


import pandas as pd

df9 = pd.DataFrame([
['1', 'Fares', 32, 'NaN'],
['2', 'Elena', 23, 'NaN'],
['NaN', 'Steven', 40, True],
['4', 'Max', 24, 'NaN'],
['5', 'Mike', 20, False],
['NaN', 'John', 40, True]])

df9.columns = ['id', 'name', 'age', 'decision']

df9


# In[6]:


df9.replace('NaN', ' ')


# ### Remove first value

# In[7]:


df = ["month", "harry styles: (Worldwide)", "zayn malik: (Worldwide)", "niall Horan: (Worldwide)", "liam payne: (Worldwide)", "louis tomlinson: (Worldwide)", "one direction: (Worldwide)"]

df


# In[8]:


df.pop(0)


# In[9]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




