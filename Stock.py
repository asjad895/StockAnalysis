import yfinance as yf
from datetime import datetime,timedelta
import os
import pandas as pd
import numpy as np
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JNJ", "JPM", "V", "WMT", "PG", "KO", "NFLX", "DIS", "XOM", "INTC", "GE", "PFE", "BABA"]
# //data Preprocessing
def fetch_merge_stock_sentiment_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)
    # //Stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    csv_file_path1 = os.path.join("pages/", "Parsed_and_Scored.csv")
    df1 = pd.read_csv(csv_file_path1)
    df1=df1.groupby('date')['sentiment_score'].mean().reset_index()
    csv_file_path2 = os.path.join("pages/", "Alpha_and_Scored.csv")
    df2 = pd.read_csv(csv_file_path2)
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    print("1____________\n",df1.shape)
    print("2_________________\n",df2.shape)
    for date in df2['date']:
        if date in df1['date'].values:
            average_score = (df2.loc[df2['date'] == date, 'sentiment_score'].values[0] +df1.loc[df1['date'] == date, 'sentiment_score'].values[0]) / 2
            df2.loc[df2['date'] == date, 'sentiment_score'] = average_score

    df2['sentiment_score'].ffill(inplace=True)
    # print(stock_data.head())
    sentiment_data=df2
    print(sentiment_data.shape)
    print(stock_data.shape)
    stock_data.index = pd.to_datetime(stock_data.index)
    # print(stock_data.head())
    # print("__________________________________________________________")
    sentiment_data = sentiment_data.set_index('date')
    sentiment_data.index=pd.to_datetime(sentiment_data.index)
    # sentiment_data=sentiment_data['sentiment_score']
    # print(sentiment_data.index)
    # print(stock_data.index)
    stock_data = stock_data.sort_index()
    sentiment_data = sentiment_data.sort_index()
    print("Preprocessed:Sentiment Data-------------------------")
    print(sentiment_data.shape)
    print(stock_data.shape)
    ls=sentiment_data.shape[0]
    stock_data['sentiment_score'] = np.nan
    stock_data.iloc[:ls, -1] = sentiment_data['sentiment_score'].values
    stock_data['sentiment_score'].ffill(inplace=True)
    # print(merged_data.shape)

    csv_file_path = os.path.join("pages/", "Stock_sent.csv")
    stock_data.to_csv(csv_file_path)

    print(f"Data for {ticker} saved to {csv_file_path}")
    print("_____________________________________________________")




# ticker = "AAPL"
# fetch_merge_stock_sentiment_data(ticker)


