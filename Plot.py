
import time
import base64
import json
import requests
import numpy as np
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import plotly.express as px
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
import plotly.graph_objects as go
nltk.downloader.download('vader_lexicon')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def plot_hourly_sentiment(parsed_and_scored_news, ticker):
    print(parsed_and_scored_news.isna().sum())
    print("hour")
    print(parsed_and_scored_news.head())
    print(parsed_and_scored_news.dtypes)
    print(parsed_and_scored_news.index)
    df = parsed_and_scored_news[['sentiment_score']].copy()
    print(df.head())
    mean_scores = df.resample('H').mean()
    # mean_scores = parsed_and_scored_news.pivot_table(index=parsed_and_scored_news.index.hour, columns=parsed_and_scored_news.index.date, values=['headline', 'neg', 'neu', 'pos', 'sentiment_score'], aggfunc='first')
    print(mean_scores)
    # mean_scores = mean_scores.dt.to_pydatetime()
    fig1 = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title=f'{ticker} Hourly Sentiment Scores',width=1000
                  ,height=600)
    # Create a column for color based on sentiment_score
    print("aaa")
    mean_scores['color'] = mean_scores['sentiment_score'].apply(lambda score: 'green' if score > 0 else 'red')
    mean_scores.dropna(subset=['sentiment_score'], inplace=True)
    # Create the line plot
    print(mean_scores)
    print("hour2")
    datetime_index = np.array(mean_scores.index)
    fig2 = px.line(mean_scores, x=datetime_index, y='sentiment_score', markers=True,color='color',
                   title=f'{ticker} Hourly Sentiment Scores (Green: Positive, Red: Negative)',width=1000
                  ,height=600)
    print("_____________________________________________________")
    return fig1, fig2

def plot_daily_sentiment(parsed_and_scored_news, ticker):
    print("daily")
    df = parsed_and_scored_news[['sentiment_score']].copy()
    print(df.head())
    mean_scores_d = df.resample('H').mean()
    fig1 = px.bar(mean_scores_d, x=mean_scores_d.index, y='sentiment_score', title=f'{ticker} Daily Sentiment Scores',width=1000
                  ,height=600)
    mean_scores_d['color'] = mean_scores_d['sentiment_score'].apply(lambda score: 'green' if score > 0 else 'red')
    mean_scores_d.dropna(subset=['sentiment_score'], inplace=True)
    fig2 = px.line(mean_scores_d, x=mean_scores_d.index, y='sentiment_score', color='color',
                   title=f'{ticker} Daily Sentiment Scores (Green: Positive, Red: Negative)',width=1000
                  ,height=600)
    print("daily2")
    print("_____________________________________________________")
    return fig1, fig2


def create_subplot_for_dataframes(dataframes,titles):
    """_summary_

    Args:
        dataframes: _10 dfs list_,titles:List

    Returns:
        Fig: _Fig of 10 stock comparative_
    """
    traces = []
    titles=titles
    for i, df in enumerate(dataframes):
        trace = go.Scatter(x=df.index, y=df['sentiment_score'], mode='lines',name=titles[i])
        traces.append(trace)
    layout = go.Layout(
        title='Sentiment Scores for 10 Stocks of Daily Article',
        width=1200, 
        height=900)
    fig = go.Figure(data=traces, layout=layout)
    print("_____________________________________________________")
    
    return fig
