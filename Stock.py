import yfinance as yf
from datetime import datetime,timedelta
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JNJ", "JPM", "V", "WMT", "PG", "KO", "NFLX", "DIS", "XOM", "INTC", "GE", "PFE", "BABA"]
end_date=datetime.now()
start_date = end_date - timedelta(days=30)
print(end_date.date())
stock_data = yf.download("AAPL", start=start_date, end=end_date, progress=False)
print(stock_data.shape)
print(stock_data.head())
