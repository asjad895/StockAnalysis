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
from Plot import plot_daily_sentiment,plot_hourly_sentiment,create_subplot_for_dataframes
from Stock import fetch_merge_stock_sentiment_data
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


def analyze_summary(business_days):
    business_days_stats = business_days
    count = business_days_stats.loc['count', 'sentiment_score']
    mean = business_days_stats.loc['mean', 'sentiment_score']
    std = business_days_stats.loc['std', 'sentiment_score']
    min_score = business_days_stats.loc['min', 'sentiment_score']
    max_score = business_days_stats.loc['max', 'sentiment_score']
    median = business_days_stats.loc['50%', 'sentiment_score']

    st.subheader('Summary Statistics')
    st.write(f"Number of Business Days: {count}")
    st.write(f"Mean Sentiment Score: {mean}")
    st.write(f"Standard Deviation of Sentiment Scores: {std}")
    st.write(f"Minimum Sentiment Score: {min_score}")
    st.write(f"Maximum Sentiment Score: {max_score}")
    st.write(f"Median Sentiment Score: {median}")

    # Box Plot
    st.write('Box Plot of Sentiment Scores')
    fig_box = px.box(business_days, y='sentiment_score')
    st.plotly_chart(fig_box)

    # Distribution Plot
    st.write('Distribution Plot of Sentiment Scores(Heatmap)')
    fig_dist = px.density_heatmap(business_days, x='sentiment_score')
    st.plotly_chart(fig_dist)

    # Summary and Suggestions
    st.write('Summary and Suggestions')
    if mean > 0:
        st.write("The mean sentiment score suggests an overall positive sentiment.")
    else:
        st.write("The mean sentiment score suggests an overall negative sentiment.")

    if std > 0.5:
        st.write("The high standard deviation indicates significant variability in sentiment, which may signal market volatility.")
    else:
        st.write("The low standard deviation suggests relatively stable sentiment.")

    if median > 0.5:
        st.write("The median sentiment score is relatively high, indicating a generally positive sentiment trend.")
    else:
        st.write("The median sentiment score is relatively low, indicating a generally negative sentiment trend.")

    st.write("Further analysis and context are needed to make specific stock market decisions.")



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
    parse_news_df, company_intro = get_news_df(ticker)
    st.info(f'{ticker} means {company_intro}')
    parsed_and_scored_news, most_negative_day, lowest_avg_sentiment, most_positive_day, highest_avg_sentiment, most_negative_week, \
        most_positive_week, lowest_avg_sentimentw, highest_avg_sentimentw = score_news(parse_news_df)
    business_days_stats, working_days_stats, holidays_stats, correlation_matrix = calculate_statistics(parsed_and_scored_news)
    fetch_merge_stock_sentiment_data(ticker=ticker)
    st.success("Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.header(":blue[Business Days Statistics:]")
        st.write("'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'")
        st.write(business_days_stats)
        analyze_summary(business_days_stats)
        st.header(":blue[Holidays Statistics:]")
        st.write("Saturday,Sunday")
        st.write(holidays_stats)
        analyze_summary(holidays_stats)
    with col2:
        st.header(":blue[Working Days Statistics:]")
        st.write("'Monday', 'Tuesday', 'Wednesday', 'Thursday'")
        st.write(working_days_stats)
        analyze_summary(working_days_stats)
        
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
    st.subheader(":blue[Comparison of Stocks]")

    op=st.button('Compare',type='primary')
    if op:
        all_df,titles=compare(ticker)
        st.subheader("Compare sentiment scores for different tickers or companies, enabling you to spot relative sentiment trends and make more informed investment choices")
        fig=create_subplot_for_dataframes(all_df,titles)
        with st.spinner('In Progress...'):
            time.sleep(10)
            st.plotly_chart(fig)
    st.success(f"Sentiment Score of {ticker} Stock Article Dataframe that we got in realtime.it is subject to chnage every time you refresh tab")
    description = f"The above chart averages the sentiment scores of {ticker} stock hourly and daily. " \
                  "The table below gives each of the most recent headlines of the stock and the negative, " \
                  "neutral, positive, and an aggregated sentiment score. " 
    st.success(description)
    st.write(parsed_and_scored_news)
    
st.info(f"Disclaimer: this project/article is not intended to provide financial, trading, and investment advice." \
        "No warranties are made regarding the accuracy of the models. Audiences should conduct their due diligence" \
        "before making any investment decisions using the methods or code presented in this article.")
    
