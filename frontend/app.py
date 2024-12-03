# frontend/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import yfinance as yf

def main():
    st.title("Financial Market Predictor")
    
    # Sidebar for inputs
    st.sidebar.title("Parameters")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
    
    # Download latest data
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")
    
    # Display stock price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close']
    ))
    st.plotly_chart(fig)
    
    if st.button("Predict Next Day Returns"):
        # Prepare data for prediction
        data = {
            "symbol": symbol,
            "market_data": hist.tail(30).to_dict(),
            "news_data": get_latest_news(symbol),
            "social_data": get_latest_tweets(symbol)
        }
        
        # Make prediction
        response = requests.post(
            "http://localhost:8000/predict",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            st.write(f"Expected Return: {result['expected_return']:.2%}")
            st.write(f"Uncertainty: {result['uncertainty']:.2%}")
            st.write("95% Confidence Interval:")
            st.write(f"Lower: {result['confidence_interval'][0]:.2%}")
            st.write(f"Upper: {result['confidence_interval'][1]:.2%}")

if __name__ == "__main__":
    main()