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
import random
nltk.downloader.download('vader_lexicon')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def preprocess_datetime(datetime_str):
    if "Today" in datetime_str:
        today = datetime.today()
        time_str = datetime_str.replace("Today", "").strip()
        return today.strftime("%Y-%m-%d") + " " + time_str
    if "Sep" in datetime_str.lower():
        print("date k lye aya")
    if "Oct" in datetime_str.lower():
        datetime_str = datetime_str.replace("oct", "Oct")
        
    
    return datetime_str

def convert_to_numeric_date(date_str):
    if date_str.startswith('20'):
        return date_str  

    date_obj = datetime.strptime(date_str, '%b-%d-%y')
    numeric_date = date_obj.strftime('%Y-%m-%d')
    return numeric_date

def get_news_df(tickers):
    """_This function will give company intro and news article for related to ticker_

    Args:
        tickers (string): Stock tickers on finviz webpage

    Returns:
        pandas df,intro:string: extracted df of new article for ticker and intro
    """
    url = f'https://finviz.com/quote.ashx?t={tickers}'
    print(url)
    req = Request(url=url,
                  headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
    response = urlopen(req)
    if response.status == 400:
            print(f"Error: {tickers} and {url}")
    soup = BeautifulSoup(response, 'html.parser')
    news_articles = soup.find_all('tr', class_='cursor-pointer has-label')
    rows = soup.find_all('tr', class_='table-light3-row')
    extracted_text = ""
    for row in rows:
        div = row.find('div')
        if div:
            div_text = div.text.strip() 
            extracted_text += div_text + "\n"
    print(extracted_text)
    dates = []
    times = []
    headlines = []
    for article in news_articles:
        time_td = article.find('td', align='right')
        publication_time = time_td.get_text(strip=True)
        publication_datetime = preprocess_datetime(publication_time)
        headline_a = article.find('td', align='left').find('a', class_='tab-link-news')
        headline = headline_a.get_text(strip=True)
        date_time_parts = publication_datetime.split()

        if len(date_time_parts) == 2:
            date, time = date_time_parts
        else:
            date = None
            time = date_time_parts[0]
        dates.append(date)
        times.append(time)
        headlines.append(headline)
        # print(headline)
    parsed_news_df = pd.DataFrame({
        'date': dates,
        'time': times,
        'headline': headlines
    })
    print(parsed_news_df['date'])
    # Filling missing dates with the last known date
    parsed_news_df['date'].ffill(inplace=True)
    print("yesy2")
    print(parsed_news_df.columns)
    print(parsed_news_df['date'])
    print(parsed_news_df.head())
    print(parsed_news_df.isna().sum())
    parsed_news_df['date'] = parsed_news_df['date'].apply(convert_to_numeric_date)
    print(parsed_news_df.head())
    parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
    print("yesy6")
    parsed_news_df['date'] = parsed_news_df['datetime'].dt.date
    csv_file_path = "parsed_news.csv"
    parsed_news_df.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to {csv_file_path}")
    print("yes")

    return parsed_news_df,extracted_text


def calculate_statistics(parse_news_df):
    business_days = parse_news_df[parse_news_df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    working_days = parse_news_df[parse_news_df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])]
    holidays = parse_news_df[parse_news_df['day_of_week'].isin(['Saturday', 'Sunday'])]

    business_days_stats = business_days.describe()
    working_days_stats = working_days.describe()
    holidays_stats = holidays.describe()

    correlation_matrix = parse_news_df[['neg', 'neu', 'pos', 'sentiment_score']].corr()

    return business_days_stats, working_days_stats, holidays_stats, correlation_matrix



def score_news(parsed_news_df):
    """_summary_

    Args:
        parsed_news_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    vader = SentimentIntensityAnalyzer()
    print(vader)
    print(parsed_news_df.head())
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    print(scores[0])
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    print("yes3")
    parsed_and_scored_news['day_of_week'] = parsed_and_scored_news['datetime'].dt.day_name()
    print("yes4")
    parsed_and_scored_news['datetime']=pd.to_datetime(parsed_and_scored_news['datetime'])
    print("yesy5")
    parsed_and_scored_news['week'] = parsed_and_scored_news['datetime'].dt.strftime('%U').astype(int) + 1
    print("sent")
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    print("sent2")
    print(parsed_and_scored_news.columns)
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], axis=1)
    print("sent3")
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})
    day_of_week_avg_sentiment = parsed_and_scored_news.groupby('day_of_week')['sentiment_score'].mean()
    print(day_of_week_avg_sentiment.head())
    print(parsed_and_scored_news.columns)
    most_negative_day = day_of_week_avg_sentiment.idxmin()

    lowest_avg_sentiment = day_of_week_avg_sentiment.min()
    most_positive_day = day_of_week_avg_sentiment.idxmax()
    highest_avg_sentiment = day_of_week_avg_sentiment.max()

    print(
        f"The day of the week with the most negative sentiment is {most_negative_day} with an average score of {lowest_avg_sentiment:.4f}")
    print(parsed_and_scored_news.head())
    print(parsed_and_scored_news.dtypes)

    week_avg_sentiment = parsed_and_scored_news.groupby('week')['sentiment_score'].mean()
    most_negative_week = week_avg_sentiment.idxmin()
    lowest_avg_sentimentw = week_avg_sentiment.min()
    most_positive_week = week_avg_sentiment.idxmax()
    highest_avg_sentimentw = week_avg_sentiment.max()

    print(f"The week with the most negative sentiment is Week {most_negative_week} with an average score of {lowest_avg_sentimentw:.4f}")
    print(f"The week with the most positive sentiment is Week {most_positive_week} with an average score of {highest_avg_sentimentw:.4f}")
    print(week_avg_sentiment)
    return parsed_and_scored_news,most_negative_day,lowest_avg_sentiment,most_positive_day,highest_avg_sentiment,most_negative_week,\
           most_positive_week,lowest_avg_sentimentw,highest_avg_sentimentw


ticker_symbols2 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JNJ", "JPM", "V",
    "WMT", "PG", "KO", "NFLX", "DIS", "XOM", "INTC", "GE", "PFE", "BABA"]
def compare(ticker):
    """_summary_

    Args:
        parsed_news_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    random_tickers = random.sample(ticker_symbols2, 9)
    dfs=[]
    j=0
    for i in random_tickers:
        j+=1
        if i==ticker:
            j+=2
        if j==10:
            break
        df=f"{i}"
        df,a=get_news_df(i)
        vader = SentimentIntensityAnalyzer()
        scores = df['headline'].apply(vader.polarity_scores).tolist()
        scores_df = pd.DataFrame(scores)
        df = df.join(scores_df, rsuffix='_right')
        df = df.rename(columns={"compound": "sentiment_score"})
        df=df.groupby('date')['sentiment_score'].mean().reset_index()
        df.set_index('date', inplace=True)
        print(df)
        dfs.append(df)
        # csv_file_path = f"{i}"+"Scored.csv"
        # df.to_csv(csv_file_path, index=False)
        # print(f"DataFrame saved to {csv_file_path}")
        print("_______________________________________________________________________________________\n",i)

    return dfs,random_tickers