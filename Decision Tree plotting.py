#!/usr/bin/env python
# coding: utf-8

# ## Sk learn version(2.1)

# ## Plotting a decision Tree

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.datasets import load_iris
from sklearn import tree


# In[5]:


cls=tree.DecisionTreeClassifier()
iris=load_iris()


# In[6]:


iris


# In[8]:


clf=cls.fit(iris.data,iris.target)


# In[9]:


clf


# In[11]:


plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)


# In[12]:


print(tree.export_text(clf))


# In[ ]:




