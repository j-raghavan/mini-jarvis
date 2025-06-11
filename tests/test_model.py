import pytest
import torch
import numpy as np
from ..models.weather_classifier import (
    WeatherIntentClassifier,
    AdvancedWeatherClassifier,
    create_model,
    PositionalEncoding
)

@pytest.fixture
def vocab_size():
    return 1000

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_length():
    return 20

@pytest.fixture
def model(vocab_size):
    return WeatherIntentClassifier(
        vocab_size=vocab_size,
        embed_dim=64,  # Smaller for testing
        hidden_dim=128,
        num_intents=6,
        num_entities=3
    )

@pytest.fixture
def advanced_model(vocab_size):
    return AdvancedWeatherClassifier(
        vocab_size=vocab_size,
        embed_dim=64,  # Smaller for testing
        hidden_dim=128,
        num_intents=6,
        num_entities=3
    )

def test_model_initialization(model):
    """Test model initialization and parameter counts"""
    # Check if model is initialized correctly
    assert isinstance(model, WeatherIntentClassifier)
    
    # Check if all components are initialized
    assert hasattr(model, 'embedding')
    assert hasattr(model, 'pos_encoding')
    assert hasattr(model, 'lstm')
    assert hasattr(model, 'attention')
    assert hasattr(model, 'intent_classifier')
    assert hasattr(model, 'entity_classifier')
    
    # Check parameter counts
    total_params = model.count_parameters()
    assert total_params > 0
    print(f"Model has {total_params:,} parameters")

def test_positional_encoding():
    """Test positional encoding functionality"""
    embed_dim = 64
    max_len = 100
    pe = PositionalEncoding(embed_dim, max_len)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Apply positional encoding
    output = pe(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, embed_dim)
    
    # Check if positional information is added
    assert not torch.allclose(output, x)

def test_model_forward_pass(model, batch_size, seq_length, vocab_size):
    """Test model forward pass"""
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    intent_logits, entity_logits = model(x)
    
    # Check output shapes
    assert intent_logits.shape == (batch_size, 6)  # num_intents = 6
    assert entity_logits.shape == (batch_size, seq_length, 3)  # num_entities = 3
    
    # Check if logits are valid
    assert not torch.isnan(intent_logits).any()
    assert not torch.isnan(entity_logits).any()

def test_model_attention(model, batch_size, seq_length, vocab_size):
    """Test attention mechanism"""
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass with attention
    intent_logits, entity_logits, attention_weights = model(x, return_attention=True)
    
    # Check attention weights
    assert attention_weights.shape == (batch_size, seq_length, seq_length)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_length))

def test_advanced_model(advanced_model, batch_size, seq_length, vocab_size):
    """Test advanced model with transformer architecture"""
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    intent_logits, entity_logits = advanced_model(x)
    
    # Check output shapes
    assert intent_logits.shape == (batch_size, 6)
    assert entity_logits.shape == (batch_size, seq_length, 3)
    
    # Check transformer attention
    intent_logits, entity_logits, attention_weights = advanced_model(x, return_attention=True)
    assert attention_weights is not None

def test_model_factory(vocab_size):
    """Test model factory function"""
    # Test basic model creation
    basic_model = create_model(vocab_size, model_type='basic')
    assert isinstance(basic_model, WeatherIntentClassifier)
    
    # Test advanced model creation
    advanced_model = create_model(vocab_size, model_type='advanced')
    assert isinstance(advanced_model, AdvancedWeatherClassifier)
    
    # Test with custom parameters
    custom_model = create_model(
        vocab_size,
        model_type='basic',
        embed_dim=128,
        hidden_dim=256,
        dropout=0.5
    )
    assert custom_model.embedding.embedding_dim == 128

def test_model_gradient_flow(model, batch_size, seq_length, vocab_size):
    """Test if gradients flow properly through the model"""
    # Create dummy input and target
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    intent_target = torch.randint(0, 6, (batch_size,))
    
    # Forward pass
    intent_logits, _ = model(x)
    
    # Calculate loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(intent_logits, intent_target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

def test_model_device_transfer(model):
    """Test model transfer between devices"""
    if torch.cuda.is_available():
        # Move to GPU
        model = model.cuda()
        assert next(model.parameters()).is_cuda
        
        # Move back to CPU
        model = model.cpu()
        assert not next(model.parameters()).is_cuda

def test_model_save_load(tmp_path, model, batch_size, seq_length, vocab_size):
    """Test model saving and loading"""
    # Save model
    save_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), save_path)
    
    # Create new model instance
    loaded_model = WeatherIntentClassifier(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_intents=6,
        num_entities=3
    )
    
    # Load state
    loaded_model.load_state_dict(torch.load(save_path))
    
    # Test if models produce same output
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    with torch.no_grad():
        original_output = model(x)
        loaded_output = loaded_model(x)
    
    assert torch.allclose(original_output[0], loaded_output[0])  # intent logits
    assert torch.allclose(original_output[1], loaded_output[1])  # entity logits 