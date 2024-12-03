# src/features/preprocessor.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
import os

class EnhancedDataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.bert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.bert_model = AutoModel.from_pretrained('ProsusAI/finbert')
        self.scaler = StandardScaler()
        
    def process_market_data(self, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """Process market data with enhanced features"""
        try:
            # Load price data and check columns
            df = pd.read_csv(f'data/raw/{symbol}_price.csv')
            logging.info(f"Columns in raw data: {df.columns.tolist()}")
            
            # Assuming the date column might have a different name
            date_column = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
            df = df.rename(columns={date_column: 'Date'})
            
            # Convert to datetime and set index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Basic features
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Rest of the processing remains the same
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Technical indicators
            self._add_technical_indicators(df)
            
            # Volume features
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_STD'] = df['Volume'].rolling(window=20).std()
            
            # Price differences
            df['HL_Diff'] = df['High'] - df['Low']
            df['OC_Diff'] = df['Open'] - df['Close']
            
            # Drop missing values
            df.dropna(inplace=True)
            
            # Scale features
            feature_cols = [col for col in df.columns if col not in ['Date']]
            df_scaled = pd.DataFrame(
                self.scaler.fit_transform(df[feature_cols]),
                columns=feature_cols,
                index=df.index
            )
            
            feature_importance = self._calculate_feature_importance(df_scaled)
            
            return df_scaled, feature_importance
            
        except Exception as e:
            logging.error(f"Error processing market data for {symbol}: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame):
        """Add technical analysis indicators"""
        # Trend
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
        
        # Momentum
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Volume
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict:
        """Calculate feature importance using correlation with returns"""
        correlations = df.corr()['Returns'].abs().sort_values(ascending=False)
        return correlations.to_dict()

    def process_news_data(self, symbol: str) -> pd.DataFrame:
        """Process news data with sentiment analysis"""
        try:
            df_news = pd.read_csv(f'data/raw/{symbol}_news.csv')
            logging.info(f"News data columns: {df_news.columns.tolist()}")
            
            # Find date/time column
            date_column = [col for col in df_news.columns if 'date' in col.lower() or 'time' in col.lower() or 'published' in col.lower()][0]
            df_news['date'] = pd.to_datetime(df_news[date_column])
            
            # Find text column for sentiment analysis
            text_column = [col for col in df_news.columns if 'title' in col.lower() or 'text' in col.lower() or 'content' in col.lower()][0]
            
            # Get BERT embeddings and sentiment
            embeddings = []
            sentiments = []
            
            for text in df_news[text_column].fillna(''):
                if isinstance(text, str) and text.strip():
                    inputs = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # Get embedding
                    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                    embeddings.append(embedding)
                    
                    # Simple sentiment score
                    sentiment = outputs.last_hidden_state.mean().item()
                    sentiments.append(sentiment)
                else:
                    embeddings.append(np.zeros((1, 768)))  # BERT base hidden size
                    sentiments.append(0.0)
            
            df_news['embedding'] = embeddings
            df_news['sentiment'] = sentiments
            
            # Aggregate by date
            daily_sentiment = df_news.groupby(df_news['date'].dt.date)['sentiment'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            return daily_sentiment
            
        except Exception as e:
            logging.error(f"Error processing news data for {symbol}: {str(e)}")
            raise