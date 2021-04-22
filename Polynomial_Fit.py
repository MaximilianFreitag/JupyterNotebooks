#!/usr/bin/env python
# coding: utf-8

# In[1]:




import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()


from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors.execute import CellExecutionError
src_notebook = nbformat.reads(Polynomial_Fit-Copy1.read(), as_version=4)   #where ff is file opened with some open("path to notebook file")

ep = ExecutePreprocessor(timeout=50, kernel_name='python3')
ep.preprocess(src_notebook, {})
html_exporter = HTMLExporter()
html_exporter.template_file = 'basic'  #basic will skip generating body and html tags.... use "all" to gen all..
(body, resources) = html_exporter.from_notebook_node(src_notebook)
print(body)   #body have html output


# In[ ]:




