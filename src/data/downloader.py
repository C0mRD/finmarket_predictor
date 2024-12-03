# src/data/downloader.py
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Dict, Union
import logging
import yaml
import json
from pathlib import Path

class DataDownloader:
    def __init__(self, config: Dict):
        self.alpha_vantage_key = config['alpha_vantage_key']
        self.newsapi_key = config['newsapi_key']
        self.symbols = config['symbols']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        
        self.setup_apis()
        
    def setup_apis(self):
        # Initialize API clients
        self.ts = TimeSeries(key=self.alpha_vantage_key)
        self.newsapi = NewsApiClient(api_key=self.newsapi_key)
        
    def download_market_data(self):
        """Download market data for all symbols"""
        for symbol in self.symbols:
            # Download from Yahoo Finance
            df_yf = yf.download(symbol, start=self.start_date, end=self.end_date)
            if df_yf.empty:
                logging.warning(f"No data found for {symbol} in the specified date range")
                continue
            
            # Download order book data from Alpha Vantage
            data_ts, _ = self.ts.get_intraday(symbol=symbol, interval='1min')
            df_av = pd.DataFrame.from_dict(data_ts, orient='index')
            
            # Save data
            df_yf.to_csv(f'data/raw/{symbol}_price.csv')
            df_av.to_csv(f'data/raw/{symbol}_orderbook.csv')
            
    def download_news_data(self):
        """Download news articles for all symbols"""
        for symbol in self.symbols:
            articles = self.newsapi.get_everything(
                q=symbol,
                from_param=self.start_date,
                to=self.end_date,
                language='en',
                sort_by='publishedAt'
            )
            
            df_news = pd.DataFrame(articles['articles'])
            df_news.to_csv(f'data/raw/{symbol}_news.csv')
            
    def download_fundamentals(self):
        """Download fundamental data for all symbols"""
        for symbol in self.symbols:
            stock = yf.Ticker(symbol)
            fundamentals = {
                'info': stock.info,
                'financials': stock.financials.to_dict(),
                'balance_sheet': stock.balance_sheet.to_dict(),
                'cashflow': stock.cashflow.to_dict()
            }
            
            pd.DataFrame(fundamentals).to_csv(f'data/raw/{symbol}_fundamentals.csv')
            
    def download_all(self):
        """Download all data types"""
        self.download_market_data()
        self.download_news_data()
        self.download_fundamentals()

if __name__ == "__main__":
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    downloader = DataDownloader(config)
    downloader.download_all()