#!/usr/bin/env python
# coding: utf-8

# ### Plotting Color graphs

# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


x = [1,20,30,4,15,1,13,4,17]
y = [30,10,3,16,3,17,5,5,18]

plt.scatter(x, y, c='red')

x2 = [5,16,28,24,21,16]
y2 = [1,13,27,12,13,6]

plt.scatter(x2, y2, c='darkviolet')

x3 = [6,17,27,8,13,7]
y3 = [23,14,2,15,11,7]

plt.scatter(x3, y3, c='mediumaquamarine')

plt.title('Correlation between Chicago & Global 5 Year MA')
plt.xlabel('Chicago 5 Year MA')
plt.ylabel('Global 5 Year MA')

plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,1,1,2,10,2,1,1,1,1]
line, = ax.plot(x, y)

ymax = max(y)
xpos = y.index(ymax)
xmax = x[xpos]

#Labeling the graph (ymax+1 is defining the distance from the word to the point)   
ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+1))

ax.set_ylim(0,20)
plt.show()


# In[2]:


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2,8, num=301)
y = np.sinc((x-2.21)*3)


fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(x,y)


ax.set_ylim(-0.3,1.5)
plt.show()


# In[ ]:




