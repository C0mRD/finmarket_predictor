import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from features.preprocessor import EnhancedDataPreprocessor
from utils.visualization import plot_price_with_indicators, plot_correlation_matrix
import yaml
import logging
import os

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = EnhancedDataPreprocessor(config)
    
    # Create processed data directory
    os.makedirs('data/processed', exist_ok=True)
    
    for symbol in config['symbols']:
        logging.info(f"Processing data for {symbol}")
        
        # Process market data
        df_market, feature_importance = preprocessor.process_market_data(symbol)
        
        # Process news data
        df_news = preprocessor.process_news_data(symbol)
        
        # Save processed data
        df_market.to_parquet(f'data/processed/{symbol}_market.parquet')
        df_news.to_parquet(f'data/processed/{symbol}_news.parquet')
        
        # Create visualizations
        fig_price = plot_price_with_indicators(df_market, symbol)
        fig_price.write_html(f'data/processed/{symbol}_price_indicators.html')
        
        fig_corr = plot_correlation_matrix(df_market)
        fig_corr.savefig(f'data/processed/{symbol}_correlation.png')
        
        logging.info(f"Top 5 important features for {symbol}:")
        for feature, importance in list(feature_importance.items())[:5]:
            logging.info(f"{feature}: {importance:.3f}")

if __name__ == "__main__":
    main() 