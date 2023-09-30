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
nltk.downloader.download('vader_lexicon')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from StatCompare import preprocess_datetime,convert_to_numeric_date,get_news_df,calculate_statistics,score_news,compare
from Plot import plot_daily_sentiment,plot_hourly_sentiment
# Streamlit app
st.set_page_config(page_title="MarketMoodMeter", page_icon="random", layout="wide", initial_sidebar_state="expanded")
# Define a function for adding a background image
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#         st.markdown(
#             f"""
#             <style>
#             .stApp {{
#             background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#             background-size: cover;
#             }}
#             </style> """, unsafe_allow_html=True
#         )

# # Call the function to set the background image
# add_bg_from_local('bg2.PNG')
st.header("MarketMoodMeter :dollar:")
ticker_symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK.B", "JNJ", "JPM", "V",
    "WMT", "PG", "KO", "NFLX", "DIS", "XOM", "INTC", "GE", "PFE", "BABA","Other"]

st.write("""In today's fast-paced and ever-evolving financial landscape, staying ahead of market sentiment is essential
        for informed decision-making. Our comprehensive Sentiment Analysis Toolkit offers a wide array of analytical tools 
        designed to empower businesses and traders alike. By harnessing the power of sentiment analysis, we provide you with
        valuable insights into the market's emotional pulse,helping you make more informed and profitable trading decisions.""")
ticker = st.selectbox('Enter Stock Ticker', ticker_symbols).upper()
print(ticker)
if ticker=='OTHER':
    ticker=st.text_input('Enter Stock Ticker name', '').upper()

if ticker:
    all_df=compare(ticker)
    parse_news_df, company_intro = get_news_df(ticker)
    st.info(f'{ticker} means {company_intro}')
    parsed_and_scored_news, most_negative_day, lowest_avg_sentiment, most_positive_day, highest_avg_sentiment, most_negative_week, \
        most_positive_week, lowest_avg_sentimentw, highest_avg_sentimentw = score_news(parse_news_df)
    # Call the function and get the results
    business_days_stats, working_days_stats, holidays_stats, correlation_matrix = calculate_statistics(parsed_and_scored_news)
    st.success("Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.header(":blue[Business Days Statistics:]")
        st.write(business_days_stats)
        st.header(":blue[Holidays Statistics:]")
        st.write(holidays_stats)
    with col2:
        st.header(":blue[Working Days Statistics:]")
        st.write(working_days_stats)
        st.header(":blue[Correlation Matrix:]")
        st.write(correlation_matrix)
    st.success(f"Hourly and Daily Sentiment of {ticker} Stock")
    print("aage aa")
    print("aur aage aa")
    fig_hourly, fig_hourly_l = plot_hourly_sentiment(parsed_and_scored_news, ticker)
    fig_daily, fig_daily_l = plot_daily_sentiment(parsed_and_scored_news, ticker)

    if st.checkbox("Hourly Analysis :bar_chart:"):
        st.balloons()
        st.plotly_chart(fig_hourly)
        st.plotly_chart(fig_hourly_l)
        st.balloons()

    if st.checkbox("Daily Analysis :bar_chart:"):
        st.plotly_chart(fig_daily)
        st.plotly_chart(fig_daily_l)
        st.balloons()

    with st.spinner("Analyzing..."):
        time.sleep(5)

    st.balloons()
    # Display the results
    col3, col4 = st.columns(2)
    with col3:
        st.header("Day of the Week Analysis")
        st.write(f"The day of the week with the most negative sentiment is {most_negative_day} with an average score of {lowest_avg_sentiment:.4f}")
        st.write(
            f"The day of the week with the most positive sentiment is {most_positive_day} with an average score of {highest_avg_sentiment:.4f}")
    with col4:
        st.header("Week Analysis")
        st.write(
            f"The week with the most negative sentiment is Week {most_negative_week} with an average score of {lowest_avg_sentimentw:.4f}")
        st.write(
            f"The week with the most positive sentiment is Week {most_positive_week} with an average score of {highest_avg_sentimentw:.4f}")

    st.success(f"Hourly and Daily Sentiment of {ticker} Stock")
    description = f"The above chart averages the sentiment scores of {ticker} stock hourly and daily. " \
                  "The table below gives each of the most recent headlines of the stock and the negative, " \
                  "neutral, positive, and an aggregated sentiment score. " \
                  "The news headlines are obtained from the FinViz website. " \
                  "Sentiments are given by the nltk.sentiment.vader Python library."
    st.success(description)
    st.table(parsed_and_scored_news)

