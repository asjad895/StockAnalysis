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
    # Check if the date starts with '20' (or any other specific year)
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
    # Find all <tr> elements representing news articles
    news_articles = soup.find_all('tr', class_='cursor-pointer has-label')
    # Find all <tr> tags with class="table-light3-row"
    rows = soup.find_all('tr', class_='table-light3-row')
    # Extract text under the <div> tag within each <tr>
    extracted_text = ""
    for row in rows:
        div = row.find('div')
        if div:
            div_text = div.text.strip()  # Remove leading/trailing whitespace
            extracted_text += div_text + "\n"
    print(extracted_text)
    # Initialize lists to store extracted data
    dates = []
    times = []
    headlines = []

    # Iterate through the news articles and extract information
    for article in news_articles:
        # Extract publication time (inside the first <td>)
        time_td = article.find('td', align='right')
        publication_time = time_td.get_text(strip=True)
        # Preprocess the publication date and time
        publication_datetime = preprocess_datetime(publication_time)
        # Extract headline (inside the <a> element within the second <td>)
        headline_a = article.find('td', align='left').find('a', class_='tab-link-news')
        headline = headline_a.get_text(strip=True)
        # Extract publication date and time
        date_time_parts = publication_datetime.split()

        if len(date_time_parts) == 2:
            date, time = date_time_parts
        else:
            date = None
            time = date_time_parts[0]

        # Append extracted data to lists
        dates.append(date)
        times.append(time)
        headlines.append(headline)
        print(headline)

    # Create a DataFrame
    parsed_news_df = pd.DataFrame({
        'date': dates,
        'time': times,
        'headline': headlines
    })
    print(parsed_news_df['date'])
    # Fill missing dates with the last known date
    parsed_news_df['date'].ffill(inplace=True)
    print("yesy2")
    print(parsed_news_df.columns)
    print(parsed_news_df['date'])
    print(parsed_news_df.head())
    print(parsed_news_df.isna().sum())
# Apply the function to the 'date' column
    parsed_news_df['date'] = parsed_news_df['date'].apply(convert_to_numeric_date)
    print(parsed_news_df.head())
    # Combine 'date' and 'time' columns into 'datetime'
    parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
    print("yesy6")
    parsed_news_df['date'] = parsed_news_df['datetime'].dt.date

    # Specify the file path where you want to save the CSV file
    csv_file_path = "parsed_news.csv"

    # Use the to_csv method to save the DataFrame to a CSV file
    parsed_news_df.to_csv(csv_file_path, index=False)

    # Print a message to confirm the file has been saved
    print(f"DataFrame saved to {csv_file_path}")
    print("yes")

    return parsed_news_df,extracted_text


def calculate_statistics(parse_news_df):
    # Separate the DataFrame into business days, working days, and holidays
    business_days = parse_news_df[parse_news_df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    working_days = parse_news_df[parse_news_df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])]
    holidays = parse_news_df[parse_news_df['day_of_week'].isin(['Saturday', 'Sunday'])]

    # Calculate statistics for each group
    business_days_stats = business_days.describe()
    working_days_stats = working_days.describe()
    holidays_stats = holidays.describe()

    # Create a correlation matrix
    correlation_matrix = parse_news_df[['neg', 'neu', 'pos', 'sentiment_score']].corr()

    return business_days_stats, working_days_stats, holidays_stats, correlation_matrix



# Define a function for scoring news sentiment
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
    # Calculate the average sentiment score for each day of the week
    day_of_week_avg_sentiment = parsed_and_scored_news.groupby('day_of_week')['sentiment_score'].mean()
    print(day_of_week_avg_sentiment.head())
    print(parsed_and_scored_news.columns)
    # Find the day of the week with the lowest average sentiment score
    most_negative_day = day_of_week_avg_sentiment.idxmin()

    # Find the lowest average sentiment score
    lowest_avg_sentiment = day_of_week_avg_sentiment.min()
    most_positive_day = day_of_week_avg_sentiment.idxmax()
    highest_avg_sentiment = day_of_week_avg_sentiment.max()

    print(
        f"The day of the week with the most negative sentiment is {most_negative_day} with an average score of {lowest_avg_sentiment:.4f}")
    print(parsed_and_scored_news.head())
    # Extract the week number from the 'datetime' column
    print(parsed_and_scored_news.dtypes)

    # Calculate the average sentiment score for each week
    week_avg_sentiment = parsed_and_scored_news.groupby('week')['sentiment_score'].mean()

    # Find the week with the most negative sentiment score
    most_negative_week = week_avg_sentiment.idxmin()
    lowest_avg_sentimentw = week_avg_sentiment.min()

    # Find the week with the most positive sentiment score
    most_positive_week = week_avg_sentiment.idxmax()
    highest_avg_sentimentw = week_avg_sentiment.max()

    print(f"The week with the most negative sentiment is Week {most_negative_week} with an average score of {lowest_avg_sentimentw:.4f}")
    print(f"The week with the most positive sentiment is Week {most_positive_week} with an average score of {highest_avg_sentimentw:.4f}")

    # Print the results
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
    random_tickers = random.sample(ticker_symbols2, 10)
    dfs=[]
    for i in random_tickers:
        if i==ticker:
            continue
        df=f"df_{i}"
        df,a=get_news_df(i)
        vader = SentimentIntensityAnalyzer()
        scores = df['headline'].apply(vader.polarity_scores).tolist()
        scores_df = pd.DataFrame(scores)
        df = df.join(scores_df, rsuffix='_right')
        df = df.rename(columns={"compound": "sentiment_score"})
        daily=df.groupby('date')['sentiment_score'].mean()
        print(daily)
        dfs.append(df)
        csv_file_path = f"{i}"+"Scored.csv"
        df.to_csv(csv_file_path, index=False)
        print(f"DataFrame saved to {csv_file_path}")
        print("_______________________________________________________________________________________\n",i)

    return dfs