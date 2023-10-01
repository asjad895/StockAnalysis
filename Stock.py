import yfinance as yf
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JNJ", "JPM", "V", "WMT", "PG", "KO", "NFLX", "DIS", "XOM", "INTC", "GE", "PFE", "BABA"]
stock_data = yf.download(tickers, start="2022-01-01", end="2023-01-01", progress=False)
print(stock_data.head())
