import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        return x + self.pe[:, :x.size(1)]

class WeatherIntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_intents=6, num_entities=3, dropout=0.3):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Intent classification head
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_intents)
        )
        
        # Entity classification head (per token)
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_entities)  # Location, Time, None
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized WeatherIntentClassifier with {self.count_parameters():,} parameters")
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'embedding' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            return_attention: Whether to return attention weights
        Returns:
            intent_logits: Intent classification logits
            entity_logits: Entity classification logits per token
            attention_weights: Optional attention weights
        """
        # Input shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        embedded = self.pos_encoding(embedded)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim*2)
        
        # Self-attention
        attn_out, attention_weights = self.attention(
            lstm_out,  # (batch_size, seq_length, hidden_dim*2)
            lstm_out,
            lstm_out
        )
        
        # Global max pooling
        pooled = torch.max(attn_out, dim=1)[0]  # (batch_size, hidden_dim*2)
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled)  # (batch_size, num_intents)
        
        # Entity classification (per token)
        entity_logits = self.entity_classifier(lstm_out)  # (batch_size, seq_length, num_entities)
        
        if return_attention:
            return intent_logits, entity_logits, attention_weights
        
        return intent_logits, entity_logits

class AdvancedWeatherClassifier(WeatherIntentClassifier):
    """Advanced model with pre-trained embeddings and transformer architecture"""
    def __init__(self, pretrained_embeddings=None, **kwargs):
        super().__init__(**kwargs)
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, 
                freeze=False,
                padding_idx=0
            )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding.embedding_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=kwargs.get('dropout', 0.3),
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Update intent classifier input dimension
        self.intent_classifier[0] = nn.Linear(
            self.embedding.embedding_dim, 
            self.intent_classifier[0].out_features
        )
        
        # Update entity classifier input dimension
        self.entity_classifier[0] = nn.Linear(
            self.embedding.embedding_dim,
            self.entity_classifier[0].out_features
        )
        
        logger.info(f"Initialized AdvancedWeatherClassifier with {self.count_parameters():,} parameters")
    
    def forward(self, x, return_attention=False):
        # Input shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        embedded = self.pos_encoding(embedded)
        
        # Transformer encoding
        transformer_out = self.transformer(embedded)  # (batch_size, seq_length, embed_dim)
        
        # Global max pooling
        pooled = torch.max(transformer_out, dim=1)[0]  # (batch_size, embed_dim)
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled)  # (batch_size, num_intents)
        
        # Entity classification (per token)
        entity_logits = self.entity_classifier(transformer_out)  # (batch_size, seq_length, num_entities)
        
        if return_attention:
            # For transformer, we can return the attention weights from the last layer
            attention_weights = self.transformer.layers[-1].self_attn.attn
            return intent_logits, entity_logits, attention_weights
        
        return intent_logits, entity_logits

def create_model(vocab_size, model_type='basic', pretrained_embeddings=None, **kwargs):
    """
    Factory function to create model instances
    Args:
        vocab_size: Size of vocabulary
        model_type: 'basic' or 'advanced'
        pretrained_embeddings: Optional pre-trained embeddings
        **kwargs: Additional model parameters
    Returns:
        WeatherIntentClassifier or AdvancedWeatherClassifier instance
    """
    if model_type == 'advanced':
        return AdvancedWeatherClassifier(
            vocab_size=vocab_size,
            pretrained_embeddings=pretrained_embeddings,
            **kwargs
        )
    else:
        return WeatherIntentClassifier(
            vocab_size=vocab_size,
            **kwargs
        ) 