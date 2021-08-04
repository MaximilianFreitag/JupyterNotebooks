
import mplfinance as mpf
import yfinance as yf


# In[ ]:


tickers = 'APPL'


# In[ ]:


data = yf.download(tickers, period='7y', interval='10d')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='4y', interval='5d')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='2y', interval='1d')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='270d', interval='1d')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


#data = yf.download(tickers, period='1y', interval='60m')
#df = data
#mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='180d', interval='60m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='120d', interval='60m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='60d', interval='30m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='30d', interval='15m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:



data = yf.download(tickers, period='10d', interval='15m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='5d', interval='5m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='3d', interval='1m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='1d', interval='1m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='12h', interval='1m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='2h', interval='1m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='1h', interval='1m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='30m', interval='1m')
df = data
mpf.plot(df, type= 'line', volume=True)


# In[ ]:


data = yf.download(tickers, period='10m', interval='1m')
df = data
mpf.plot(df, type= 'line', volume=True)



