import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# //Data Cllection
df=pd.read_csv('pages/Stock_sent.csv')
# //Data cleaning
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
    df=merged_data
    st.subheader("Visualize the trend of sentimnet score with stock price")
    fig = go.Figure()
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stock_Open'], mode='lines', name='Open Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stock_High'], mode='lines', name='High Price'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stock_Low'], mode='lines', name='Low Price'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stock_Volume'], mode='lines', name='volume'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Sentiment_Score'], mode='lines', name='Sentiment score'), row=5, col=1)
    fig.update_xaxes(title_text='Date', row=5, col=1)
    fig.update_yaxes(title_text='Open', row=1, col=1)
    fig.update_yaxes(title_text='High ', row=2, col=1)
    fig.update_yaxes(title_text='Low ', row=3, col=1)
    fig.update_yaxes(title_text='volume ', row=4, col=1)
    fig.update_yaxes(title_text='Sentiment Score ', row=5, col=1)
    fig.update_layout(title_text='Price Time Series', showlegend=True,width=1200,height=600)
    st.plotly_chart(fig)    
analyze_data(df)

stockSent_chart(df)