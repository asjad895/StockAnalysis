import yfinance as yf
from datetime import datetime,timedelta
import os
import pandas as pd
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JNJ", "JPM", "V", "WMT", "PG", "KO", "NFLX", "DIS", "XOM", "INTC", "GE", "PFE", "BABA"]
# //data Preprocessing
def fetch_merge_stock_sentiment_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    # //Stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    csv_file_path = os.path.join("pages/", "Parsed_and_Scored.csv")
    sentiment_data = pd.read_csv(csv_file_path)
    # print(stock_data.head())
    sentiment_data=pd.read_csv(csv_file_path)
    stock_data.index = pd.to_datetime(stock_data.index)
    # print(stock_data.head())
    # print("__________________________________________________________")
    sentiment_data = sentiment_data.set_index('date')
    sentiment_data.index=pd.to_datetime(sentiment_data.index)
    sentiment_data=sentiment_data['sentiment_score']
    # print(sentiment_data.index)
    # print(stock_data.index)
    stock_data = stock_data.sort_index()
    sentiment_data = sentiment_data.sort_index()
    sentiment_data_filtered = sentiment_data[sentiment_data.index.isin(stock_data.index)]
    merged_data = pd.merge(stock_data, sentiment_data_filtered, left_index=True, right_index=True, how='inner')
    # print(merged_data.shape)

    csv_file_path = os.path.join("pages/", "Stock_sent.csv")
    merged_data.to_csv(csv_file_path)

    print(f"Data for {ticker} saved to {csv_file_path}")
    print("_____________________________________________________")




ticker = "AAPL"
fetch_merge_stock_sentiment_data(ticker)


