# src/models/multimodal_transformer.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Dict, List

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained('finbert-sentiment')
        self.proj = nn.Linear(768, hidden_dim)  # BERT hidden size is 768
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(outputs.last_hidden_state)

class MultiModalTransformer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Encoders for different modalities
        self.market_encoder = TimeSeriesEncoder(
            input_dim=config['market_features'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads']
        )
        
        self.news_encoder = TextEncoder(config['hidden_dim'])
        self.social_encoder = TextEncoder(config['hidden_dim'])
        
        # Multi-modal fusion transformer
        fusion_layer = TransformerEncoderLayer(
            config['hidden_dim'],
            config['num_heads']
        )
        self.fusion_transformer = TransformerEncoder(
            fusion_layer,
            config['num_fusion_layers']
        )
        
        # Prediction heads
        self.return_predictor = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'] // 2, 2)  # Mean and variance
        )
        
    def forward(self, market_data, news_data, social_data):
        # Encode different modalities
        market_features = self.market_encoder(market_data)
        news_features = self.news_encoder(**news_data)
        social_features = self.social_encoder(**social_data)
        
        # Concatenate features
        combined_features = torch.cat(
            [market_features, news_features, social_features],
            dim=1
        )
        
        # Multi-modal fusion
        fused_features = self.fusion_transformer(combined_features)
        
        # Get predictions
        predictions = self.return_predictor(fused_features.mean(dim=1))
        
        # Split into mean and variance predictions
        mean, log_var = predictions.chunk(2, dim=-1)
        
        return mean, log_var

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]