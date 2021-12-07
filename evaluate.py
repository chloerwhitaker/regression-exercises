#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import sklearn.metrics
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt


# ### Individual functions to caluclate residual, sse, mse, ess, tss,  and r2

# In[11]:


def residuals(actual, predicted):
    return actual - predicted


# In[3]:


def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()


# In[4]:


def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n


# In[5]:


def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))


# In[6]:


def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()


# In[7]:


def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()


# In[9]:


def r2_score(actual, predicted):
    return ess(actual, predicted) / tss(actual)


# ### Function to plt residuals

# In[12]:


def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()


# ### Function to return regression errors

# In[13]:


def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    })


# ### Function to return baseline errors

# In[14]:


def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }


# ### Function to determine if model is better than the baseline

# In[15]:


def better_than_baseline(actual, predicted):
    rmse_baseline = rmse(actual, actual.mean())
    rmse_model = rmse(actual, predicted)
    return rmse_model < rmse_baseline


# In[16]:


# to run:
# better_than_baseline(tips.tip, tips.yhat)


# In[ ]:




