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
st.header("--Analyze how market influencing Stock prices--")
st.dataframe(df)

def analyze_data(merged_data):
    correlation_matrix = merged_data.corr()
    st.subheader("Correlation Matrix Heatmap")
    st.write(correlation_matrix.style.background_gradient(cmap='coolwarm'))
    st.subheader("Insights")
    st.write("Correlation values close to 1 indicate a strong positive relationship, while values close to -1 indicate a strong negative relationship.")
    if correlation_matrix['Stock_Close']['Sentiment_Score'] > 0.5:
        st.write("Positive correlation between Sentiment Score and Stock Close Price.")
    elif correlation_matrix['Stock_Close']['Sentiment_Score'] < -0.5:
        st.write("Negative correlation between Sentiment Score and Stock Close Price.")
    else:
        st.write("Weak or no significant correlation.")
        
        
analyze_data(df)