#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd




df = pd.DataFrame([
['1', 'Fares', 32, True],
['2', 'Elena', 23, False],
['3', 'Steven', 40, True],
['4', 'Max', 20, True],
['5', 'Chris', 13, False],
['6', 'Kurt', 33, True],
['7', 'Sophia', 23, False]])

df.columns = ['id', 'name', 'age', 'decision']

df


# In[4]:


n = 5
skip_func = lambda x: x%n != 0
df = pd.(skiprows = skip_func)


# In[7]:


df2 = df2(skiprows=[0,2,5])
#print('Contents of the Dataframe created by skipping specifying lines from csv file ')
print(df2)


# In[10]:


print('Enter number here: ')
integer = input()
while not(integer.isdigit()):
    print('only numbers!')
    print('Enter number here: ')
    integer = input()


NumToLetter = {'0': 'A', '1': 'C', '2': 'G', '3': 'D', '4': 'B', '5': 'F', '6': 'V', '7': 'K', '8': 
'I', '9': 'P'}


letter = ''
integer_length = len(integer)


if integer_length > 0:
   letter = NumToLetter[integer]

print(letter)


# In[ ]:




