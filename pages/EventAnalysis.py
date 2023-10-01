
import time
import base64
import json
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import plotly.express as px
from PIL import Image
import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
nltk.downloader.download('vader_lexicon')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from StatCompare import preprocess_datetime,convert_to_numeric_date,get_news_df,calculate_statistics,score_news,compare
from Plot import plot_daily_sentiment,plot_hourly_sentiment,create_subplot_for_dataframes
df=pd.read_csv('Parsed_and_Scored.csv')
st.warning("Please load Home page fully so that data can be fetch in realtime.Thankyou")
st.subheader('Event Analysis Results')
extreme_events = df[(df['sentiment_score'] > 0.5) | (df['sentiment_score'] < -0.5)]
st.dataframe(extreme_events)
selected_event = st.selectbox('Select an event to analyze:', extreme_events['headline'].tolist())
if selected_event:
    event_row = df[df['headline'] == selected_event].iloc[0]
    print(event_row)
    st.subheader('Selected Event Content:')
    st.write(event_row['headline'])
    sentiment = event_row['sentiment_score']
    st.write(f'Sentiment Score: {sentiment}')
    st.write(f'Day of the Week: {event_row["day_of_week"]}')
    st.write(f'Week: {event_row["week"]}')
    event_headline = event_row['headline']
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(event_headline)
    st.subheader('Word Cloud Analysis for Selected Event:')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)