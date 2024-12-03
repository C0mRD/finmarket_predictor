import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt

def plot_price_with_indicators(df: pd.DataFrame, symbol: str):
    """Create an interactive plot with price and technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add technical indicators
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        name='20-day SMA',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_upper'],
        name='Bollinger Upper',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_lower'],
        name='Bollinger Lower',
        line=dict(color='gray', dash='dash'),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=f'{symbol} Price and Indicators',
        yaxis_title='Price',
        xaxis_title='Date'
    )
    
    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    """Plot correlation matrix of features"""
    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    return plt 