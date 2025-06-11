import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from ..training.trainer import WeatherDataset, WeatherTrainer
from ..models.weather_classifier import create_model

@pytest.fixture
def sample_data(tmp_path):
    """Create sample training data"""
    # Create sample data
    data = {
        'text': [
            'what is the weather like in london',
            'will it rain in paris tomorrow',
            'temperature in new york',
            'compare weather in tokyo and berlin'
        ],
        'intent': [0, 1, 2, 4],  # Using intent indices
        'entities': [
            '{"location": "london"}',
            '{"location": "paris", "time": "tomorrow"}',
            '{"location": "new york"}',
            '{"location1": "tokyo", "location2": "berlin"}'
        ]
    }
    
    # Create train and test files
    train_df = pd.DataFrame(data)
    test_df = pd.DataFrame(data)
    
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return train_path, test_path

@pytest.fixture
def vocab_size():
    return 1000

@pytest.fixture
def model(vocab_size):
    return create_model(
        vocab_size=vocab_size,
        model_type='basic',
        embed_dim=64,  # Smaller for testing
        hidden_dim=128
    )

@pytest.fixture
def trainer(model, tmp_path):
    return WeatherTrainer(
        model=model,
        device='cpu',
        model_dir=tmp_path,
        learning_rate=0.001,
        weight_decay=1e-5
    )

def test_dataset_creation(sample_data):
    """Test dataset creation and loading"""
    train_path, test_path = sample_data
    
    # Create dataset
    dataset = WeatherDataset(train_path, max_length=20)
    
    # Test dataset length
    assert len(dataset) == 4
    
    # Test item retrieval
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 3  # text, intent, entities
    
    # Test tensor shapes
    text, intent, entities = item
    assert isinstance(text, torch.Tensor)
    assert isinstance(intent, torch.Tensor)
    assert isinstance(entities, torch.Tensor)
    assert text.shape[0] <= 20  # max_length
    assert intent.shape == (1,)
    assert entities.shape[0] <= 20  # max_length

def test_dataloader_creation(sample_data, trainer):
    """Test dataloader creation"""
    train_path, test_path = sample_data
    
    # Create dataloaders
    train_loader, val_loader = trainer.prepare_dataloaders(
        train_path,
        test_path,
        batch_size=2
    )
    
    # Test dataloader properties
    assert len(train_loader) == 2  # 4 samples, batch_size=2
    assert len(val_loader) == 2
    
    # Test batch loading
    batch = next(iter(train_loader))
    assert len(batch) == 3  # text, intent, entities
    assert batch[0].shape[0] == 2  # batch_size
    assert batch[1].shape[0] == 2
    assert batch[2].shape[0] == 2

def test_trainer_initialization(trainer, model):
    """Test trainer initialization"""
    assert trainer.model == model
    assert trainer.device == 'cpu'
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    # Test loss functions
    assert isinstance(trainer.intent_criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(trainer.entity_criterion, torch.nn.CrossEntropyLoss)

def test_training_step(trainer, sample_data):
    """Test single training step"""
    train_path, _ = sample_data
    train_loader, _ = trainer.prepare_dataloaders(
        train_path,
        train_path,  # Using same data for validation
        batch_size=2
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    
    # Training step
    loss, intent_acc, entity_acc = trainer._train_epoch(train_loader)
    
    # Check metrics
    assert isinstance(loss, float)
    assert isinstance(intent_acc, float)
    assert isinstance(entity_acc, float)
    assert 0 <= intent_acc <= 1
    assert 0 <= entity_acc <= 1

def test_validation_step(trainer, sample_data):
    """Test validation step"""
    train_path, _ = sample_data
    _, val_loader = trainer.prepare_dataloaders(
        train_path,
        train_path,  # Using same data for validation
        batch_size=2
    )
    
    # Validation step
    val_loss, val_intent_acc, val_entity_acc = trainer._validate(val_loader)
    
    # Check metrics
    assert isinstance(val_loss, float)
    assert isinstance(val_intent_acc, float)
    assert isinstance(val_entity_acc, float)
    assert 0 <= val_intent_acc <= 1
    assert 0 <= val_entity_acc <= 1

def test_model_saving(trainer, model, tmp_path):
    """Test model saving and loading"""
    # Save model
    save_path = tmp_path / "model.pth"
    trainer.save_model(save_path)
    
    # Check if file exists
    assert save_path.exists()
    
    # Load model
    loaded_model = create_model(
        vocab_size=1000,
        model_type='basic',
        embed_dim=64,
        hidden_dim=128
    )
    loaded_model.load_state_dict(torch.load(save_path))
    
    # Test if models are equivalent
    x = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    with torch.no_grad():
        original_output = model(x)
        loaded_output = loaded_model(x)
    
    assert torch.allclose(original_output[0], loaded_output[0])  # intent logits
    assert torch.allclose(original_output[1], loaded_output[1])  # entity logits

def test_learning_rate_scheduling(trainer, sample_data):
    """Test learning rate scheduling"""
    train_path, _ = sample_data
    train_loader, val_loader = trainer.prepare_dataloaders(
        train_path,
        train_path,
        batch_size=2
    )
    
    # Get initial learning rate
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    
    # Train for a few epochs with high validation loss
    for _ in range(3):
        trainer._train_epoch(train_loader)
        trainer._validate(val_loader)
    
    # Check if learning rate decreased
    final_lr = trainer.optimizer.param_groups[0]['lr']
    assert final_lr < initial_lr

def test_early_stopping(trainer, sample_data):
    """Test early stopping mechanism"""
    train_path, _ = sample_data
    train_loader, val_loader = trainer.prepare_dataloaders(
        train_path,
        train_path,
        batch_size=2
    )
    
    # Train with early stopping
    best_model = trainer.train(
        train_loader,
        val_loader,
        epochs=5,
        early_stopping_patience=2
    )
    
    # Check if best model is saved
    assert best_model is not None
    assert (trainer.model_dir / "best_model.pth").exists()

def test_metrics_logging(trainer, sample_data, tmp_path):
    """Test metrics logging"""
    train_path, _ = sample_data
    train_loader, val_loader = trainer.prepare_dataloaders(
        train_path,
        train_path,
        batch_size=2
    )
    
    # Train for one epoch
    trainer.train(
        train_loader,
        val_loader,
        epochs=1,
        log_interval=1
    )
    
    # Check if metrics file exists
    metrics_file = tmp_path / "training_metrics.json"
    assert metrics_file.exists()
    
    # Check metrics content
    import json
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    assert 'train_loss' in metrics
    assert 'val_loss' in metrics
    assert 'train_intent_acc' in metrics
    assert 'val_intent_acc' in metrics

def test_error_handling(trainer):
    """Test error handling in training"""
    # Test with invalid data
    with pytest.raises(Exception):
        trainer.prepare_dataloaders(
            "nonexistent_train.csv",
            "nonexistent_test.csv",
            batch_size=2
        )
    
    # Test with invalid batch size
    with pytest.raises(Exception):
        trainer.prepare_dataloaders(
            "train.csv",
            "test.csv",
            batch_size=0
        )
    
    # Test with invalid learning rate
    with pytest.raises(ValueError):
        WeatherTrainer(
            model=trainer.model,
            device='cpu',
            model_dir=trainer.model_dir,
            learning_rate=-0.001
        ) 