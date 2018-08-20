
# coding: utf-8

# In[4]:


from pycbc.detector import Detector


# In[5]:


det_h1 = Detector('H1')
det_l1 = Detector('L1')
det_v1 = Detector('V1')


# In[6]:


end_time = 1192529720
declination = 0.65
right_ascension = 4.67
polarization = 2.34


# In[7]:


det_h1.antenna_pattern(right_ascension, declination, polarization, end_time)

