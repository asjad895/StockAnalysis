import time
import base64
import json
import requests
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import plotly
import plotly.express as px
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from streamlit_lottie import st_lottie

nltk.downloader.download('vader_lexicon')
# Streamlit app
st.set_page_config(page_title="Stock Sentiment Analysis", page_icon="random", layout="wide", initial_sidebar_state="expanded")
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

# Define a function for getting news from Finviz
from datetime import datetime
def preprocess_datetime(datetime_str):
    if "Today" in datetime_str:
        today = datetime.today()
        time_str = datetime_str.replace("Today", "").strip()
        return today.strftime("%Y-%m-%d") + " " + time_str
    return datetime_str

def get_news_df(tickers):
    url = f'https://finviz.com/quote.ashx?t={tickers}'
    print(url)
    req = Request(url=url,
                  headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
    response = urlopen(req)
    soup = BeautifulSoup(response, 'html.parser')
    # Find all <tr> elements representing news articles
    news_articles = soup.find_all('tr', class_='cursor-pointer has-label')
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
    # Fill missing dates with the last known date
    parsed_news_df['date'].fillna(method='ffill', inplace=True)
    # Combine 'date' and 'time' columns into 'datetime'
    parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
    parsed_news_df['date'] = parsed_news_df['datetime'].dt.date

    # Specify the file path where you want to save the CSV file
    csv_file_path = 'parsed_news.csv'

    # Use the to_csv method to save the DataFrame to a CSV file
    parsed_news_df.to_csv(csv_file_path, index=False)

    # Print a message to confirm the file has been saved
    print(f"DataFrame saved to {csv_file_path}")

    return parsed_news_df


# Define a function for scoring news sentiment
def score_news(parsed_news_df):
    vader = SentimentIntensityAnalyzer()
    print(vader)
    print(parsed_news_df.head())
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    print(scores[0])
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news['day_of_week'] = parsed_and_scored_news['datetime'].dt.day_name()
    parsed_and_scored_news['week'] = parsed_and_scored_news['datetime'].dt.week
    print("sent")
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    print("sent2")
    print(parsed_and_scored_news.columns)
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)
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


# Define a function for plotting hourly sentiment
def plot_hourly_sentiment(parsed_and_scored_news, ticker):
    print(parsed_and_scored_news.isna().sum())
    print("hour")
    mean_scores = parsed_and_scored_news.resample('H').mean()
    print(mean_scores)
    fig1 = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title=f'{ticker} Hourly Sentiment Scores')
    # Create a column for color based on sentiment_score
    mean_scores['color'] = mean_scores['sentiment_score'].apply(lambda score: 'green' if score > 0 else 'red')
    mean_scores.dropna(subset=['sentiment_score'], inplace=True)
    # Create the line plot
    print(mean_scores)
    print("hour2")
    fig2 = px.line(mean_scores, x=mean_scores.index, y='sentiment_score', markers=True,color='color',
                      title=f'{ticker} Hourly Sentiment Scores (Green: Positive, Red: Negative)')
    return fig1, fig2

# Define a function for plotting daily sentiment
def plot_daily_sentiment(parsed_and_scored_news, ticker):
    mean_scores_d = parsed_and_scored_news.resample('D').mean()
    print("daily")
    fig1 = px.bar(mean_scores_d, x=mean_scores_d.index, y='sentiment_score', title=f'{ticker} Daily Sentiment Scores')
    # Create a column for color based on sentiment_score
    mean_scores_d['color'] = mean_scores_d['sentiment_score'].apply(lambda score: 'green' if score > 0 else 'red')
    mean_scores_d.dropna(subset=['sentiment_score'], inplace=True)
    # Create the line plot
    fig2 = px.line(mean_scores_d, x=mean_scores_d.index, y='sentiment_score', color='color',
                   title=f'{ticker} Daily Sentiment Scores (Green: Positive, Red: Negative)')
    print("daily2")
    return fig1, fig2



st.header("Stock News Sentiment Analyzer :dollar:")
ticker = st.text_input('Enter Stock Ticker', '').upper()

if ticker:
    try:
        st.success(f"Hourly and Daily Sentiment of {ticker} Stock")
        parse_news_df = get_news_df(ticker)
        print("aage aa")
        parsed_and_scored_news,most_negative_day,lowest_avg_sentiment,most_positive_day,highest_avg_sentiment,most_negative_week,\
           most_positive_week,lowest_avg_sentimentw,highest_avg_sentimentw= score_news(parse_news_df)
        print("aur aage aa")
        fig_hourly,fig_hourly_l = plot_hourly_sentiment(parsed_and_scored_news, ticker)
        fig_daily,fig_daily_l = plot_daily_sentiment(parsed_and_scored_news, ticker)

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
        st.header("Day of the Week Analysis")
        st.write(
            f"The day of the week with the most negative sentiment is {most_negative_day} with an average score of {lowest_avg_sentiment:.4f}")
        st.write(
            f"The day of the week with the most positive sentiment is {most_positive_day} with an average score of {highest_avg_sentiment:.4f}")

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
    except Exception as e:
        st.warning("An error occurred. Please enter a correct stock ticker, e.g., 'AAPL' above and hit Enter.")
        st.info("If you want to explore a ticker, click the link below.")