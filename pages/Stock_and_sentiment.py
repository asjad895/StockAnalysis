import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
st.set_page_config(page_title="MarketMoodMeter-StockAnalytics", page_icon="random", layout="wide", initial_sidebar_state="expanded")
st.title("--Analyze how market influencing Stock prices--")
st.dataframe(df)

def analyze_data(merged_data):
    correlation_matrix = merged_data.corr()
    st.write(correlation_matrix.style.background_gradient(cmap='coolwarm'))
    st.header(":blue[Insights]")
    st.info("i have taken thresold .5/-.5 for influencing .")
    for column in correlation_matrix.columns:
        if column != 'Sentiment_Score':
            st.subheader(f"Analysis for {column} vs. Sentiment_Score")
            correlation_value = correlation_matrix[column]['Sentiment_Score']
            st.write(f"Correlation with Sentiment_Score: {correlation_value}")
            if correlation_value > 0.5:
                st.write("Positive correlation: Positive sentiment may influence stock price.")
            elif correlation_value < -0.5:
                st.write("Negative correlation: Negative sentiment may influence stock price.")
            else:
                st.write("Weak or no significant correlation.")
        
def stockSent_chart(merged_data):
    st.subheader("Visualize the trend of sentimnet score with stock price")
    traces=[]
    fig = go.Figure()
    for column in merged_data.columns:
        trace = go.Scatter(x=df.index, y=df[column], mode='lines',name=column)
        traces.append(trace)
    layout = go.Layout(
        title='Comparative charts with sentiments score',
        width=1200,
        height=900)
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig)     
analyze_data(df)

stockSent_chart(df)