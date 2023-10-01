import pandas as pd
import numpy as np
import streamlit as st
df=pd.read_csv('pages/Stock_sent.csv')
df=df.rename(columns={
        'Open': 'Stock_Open',
        'High': 'Stock_High',
        'Low': 'Stock_Low',
        'Close': 'Stock_Close',
        'Adj Close': 'Stock_Adj_Close',
        'Volume': 'Stock_Volume',
        'sentiment_score': 'Sentiment_Score',
        'Unnamed: 0':'Date'
    })
# df['Date']=pd.to_datetime(df['Date'])
df=df.set_index('Date')
df.index=pd.to_datetime(df.index)
st.dataframe(df)