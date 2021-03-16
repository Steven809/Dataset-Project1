
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[57]:


NYC = pd.read_csv("DSNY_Monthly_Tonnage_Data.csv")


# In[58]:


NYC['PAPERTONSCOLLECTED'] = NYC['PAPERTONSCOLLECTED'].replace(np.nan, 0)
NYC['MGPTONSCOLLECTED'] = NYC['MGPTONSCOLLECTED'].replace(np.nan, 0)
NYC['RESORGANICSTONS'] = NYC['RESORGANICSTONS'].replace(np.nan, 0)
NYC['SCHOOLORGANICTONS'] = NYC['SCHOOLORGANICTONS'].replace(np.nan, 0)
NYC['LEAVESORGANICTONS'] = NYC['LEAVESORGANICTONS'].replace(np.nan, 0)
NYC['XMASTREETONS'] = NYC['XMASTREETONS'].replace(np.nan, 0)


# In[55]:


NYC


# In[79]:


Month_2020 = NYC["MONTH"] > "2020"
Month_2020_3 = NYC["MONTH"] < "2020 / 03"


# In[81]:


NYC2 = NYC[Month_2020 & Month_2020_3]
NYC2


# In[86]:


NYC2.dtypes


# In[88]:


sns.pairplot(data = NYC2, hue = "BOROUGH")


# In[93]:


sns.scatterplot(x = "REFUSETONSCOLLECTED", y = "MGPTONSCOLLECTED", hue = "BOROUGH", data = NYC2)
plt.title("Recycle")


# In[98]:


sns.scatterplot(x = "REFUSETONSCOLLECTED", y = "PAPERTONSCOLLECTED", hue = "BOROUGH", data = NYC2)
plt.title("Recycle")

