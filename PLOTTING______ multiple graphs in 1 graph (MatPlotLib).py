#!/usr/bin/env python
# coding: utf-8

# ### 4 plots in 1

# In[3]:


import numpy as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')


a = [1.02, .95, .87, .77, .67, .46, .74, .60]
b = [0.39, .32, .27, .22, .18, .15, .13, .12]
c = [0.49, .42, .37, .32, .28, .35, .33, .52]
d = [0.29, .52, .47, .52, .58, .35, .43, .32]

figure(num=None, figsize=(12, 8), dpi=50, facecolor='w', edgecolor='k')



plt.plot(a, 'r') # plotting a separately 
plt.plot(b, 'b') # plotting b separately
plt.plot(c, 'g') # plotting c separately 
plt.plot(d, 'y') # plotting d separately 



plt.title('Title of your graph')
plt.xlabel('Something X')
plt.ylabel('Something Y')

   

plt.show()


# In[ ]:


fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')


# In[ ]:





# In[ ]:





# ### 4 different plots in 1

# In[4]:


import numpy as np
from numpy import e, pi, sin, exp, cos
import matplotlib.pyplot as plt


a = [1.02, .95, .87, .77, .67, .46, .74, .60]
b = [0.39, .32, .27, .22, .18, .15, .13, .12]
c = [0.49, .42, .37, .32, .28, .35, .33, .52]
d = [0.29, .52, .47, .52, .58, .35, .43, .32]


python_course_green = "#476042"
fig = plt.figure(figsize=(6, 4))

#t = np.arange(-5.0, 1.0, 0.1)

sub1 = fig.add_subplot(221) # instead of plt.subplot(2, 2, 1)
sub1.set_title('Something A') # non OOP: plt.title('The function f')
sub1.plot(a)


sub2 = fig.add_subplot(222, facecolor="lightgrey")
sub2.set_title('Something B')
sub2.plot(b)


t = np.arange(-3.0, 2.0, 0.02)
sub3 = fig.add_subplot(223)
sub3.set_title('Something C')
sub3.plot(c)

t = np.arange(-0.2, 0.2, 0.001)
sub4 = fig.add_subplot(224, facecolor="lightgrey")
sub4.set_title('Something D')
sub4.plot(d)



plt.tight_layout()
plt.show()


# In[6]:


import numpy as np 
from matplotlib import pyplot as plt 

x = np.arange(12,30) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show()


# In[ ]:




