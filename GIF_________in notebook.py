#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import HTML
HTML('<img src="../giphy.gif">')


# In[ ]:





# In[ ]:


from IPython.display import Image
Image(filename="../giphy.gif.png")


# In[ ]:





# In[ ]:


with open('../giphy.gif','rb') as f:
    display(Image(data=f.read(), format='png'))


# In[ ]:





# In[ ]:




