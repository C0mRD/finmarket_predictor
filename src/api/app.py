# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
from ..models.multimodal_transformer import MultiModalTransformer

app = FastAPI(
    title="Financial Market Predictor",
    description="Multi-modal market prediction API",
    version="1.0.0"
)

# Load model
model = MultiModalTransformer(config)  # Load config
model.load_state_dict(torch.load('models/market_predictor.pt'))
model.eval()

class PredictionRequest(BaseModel):
    symbol: str
    market_data: Dict[str, List[float]]
    news_data: Dict[str, str]
    social_data: Dict[str, str]

class PredictionResponse(BaseModel):
    expected_return: float
    uncertainty: float
    confidence_interval: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess inputs
        market_tensor = preprocess_market_data(request.market_data)
        news_tensor = preprocess_news_data(request.news_data)
        social_tensor = preprocess_social_data(request.social_data)
        
        # Get prediction
        with torch.no_grad():
            mean, log_var = model(market_tensor, news_tensor, social_tensor)
            
        std = torch.exp(0.5 * log_var)
        
        return PredictionResponse(
            expected_return=float(mean[0]),
            uncertainty=float(std[0]),
            confidence_interval=[
                float(mean[0] - 1.96 * std[0]),
                float(mean[0] + 1.96 * std[0])
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))