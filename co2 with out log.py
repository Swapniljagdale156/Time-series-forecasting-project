#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import numpy as np             #for numerical computations like log,exp,sqrt etc
import pandas as pd            #for reading & storing data, pre-processing
import matplotlib.pylab as plt #for visualization
#for making sure matplotlib plots are generated in Jupyter notebook itself
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


import pandas as pd 
from datetime import datetime


def dateparse(dates):
    return datetime.strptime(dates, '%Y')


data = pd.read_excel("CO2 dataset.xlsx",
                           parse_dates=['Year'],
                           index_col='Year',
                           date_parser=dateparse)


# In[5]:


data


# In[6]:


plt.xlabel('Years')
plt.ylabel('CO2')
plt.plot(data)


# In[7]:


rolling_mean=data.rolling(window=12).mean()
print(rolling_mean)


# In[8]:


rolling_std=data.rolling(window=12).std()
print(rolling_std)


# In[9]:


plt.plot(data,color='blue',label='given time series')
plt.plot(rolling_mean,color='orange',label='rolling mean')
plt.plot(rolling_std,color='green',label='rolling standard deviation')
plt.legend()


# In[13]:


# data_log=np.log(data)
# data.head()
# plt.plot(data_log)
data_n = data


# In[14]:


# new_rolling_mean = data_log.rolling(window=12).mean()
# data_log_subtract_mean = data_log - new_rolling_mean
new_rolling_mean = data_n.rolling(window=12).mean()
data_n_subtract_mean = data_n - new_rolling_mean


# In[16]:


data_n_subtract_mean


# In[17]:


data_n_subtract_mean.dropna(inplace=True)


# In[18]:


transformed_rolling_mean=data_n_subtract_mean.rolling(window=12).mean()
transformed_rolling_std=data_n_subtract_mean.rolling(window=24).std()


# without log

# In[19]:


plt.plot(data_n_subtract_mean,color='blue',label='given time series')
plt.plot(transformed_rolling_mean,color='red',label='rolling mean')
plt.plot(transformed_rolling_std,color='green',label='rolling standard deviation')
plt.legend()


# In[20]:


data_n_shifting= data_n-data_n.shift() 
#time shifting is performed.
plt.plot(data_n_shifting)


# In[21]:


data_n_shifting.dropna(inplace=True)


# In[89]:


plot_acf = acf(data_n_shifting, nlags=30)
plot_pacf = pacf(data_n_shifting, nlags=100, method='ols')

#code to plot acf
plt.subplot(121)
plt.plot(plot_acf)
plt.axhline(y=0, linestyle='--', color='red')
plt.axhline(y=-1.96/np.sqrt(len(data_n_shifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_n_shifting)), linestyle='--', color='gray')
plt.title('ACF')            

#code to Plot PACF
plt.subplot(122)
plt.plot(plot_pacf)
plt.axhline(y=0, linestyle='--', color='red')
plt.axhline(y=-1.96/np.sqrt(len(data_n_shifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_n_shifting)), linestyle='--', color='gray')
plt.title('PACF')
            


# In[90]:


model = ARIMA(data_n, order=(4,0,0))
results = model.fit(disp=-20)
plt.plot(data_n_shifting)
plt.plot(results.fittedvalues, color='red')


# In[118]:


arima_model_prediction = pd.Series(results.fittedvalues, copy=True)
arima_model_prediction_cumsum= arima_model_prediction.cumsum()
predictions_ARIMA_n = pd.Series(data_n['CO2'].iloc[0], index=data_n.index)
predictions_ARIMA_n = predictions_ARIMA_n.add(arima_model_prediction_cumsum, fill_value=0)
final_ARIMA_predictions= np.exp(predictions_ARIMA_n)#reversing transformaton by using exp()
plt.plot(data)
plt.plot(final_ARIMA_predictions, color = 'red')


# In[119]:


results.plot_predict(2,220)


# In[120]:


results.predict(2,225).tail(10)


# In[ ]:





# In[ ]:




