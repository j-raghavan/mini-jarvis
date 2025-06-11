import pytest
import pandas as pd
from pathlib import Path
from ..data.data_generator import WeatherDatasetGenerator, TextPreprocessor

@pytest.fixture
def generator():
    return WeatherDatasetGenerator()

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_dataset_generator_initialization(generator):
    """Test if generator initializes with correct intents"""
    assert len(generator.intents) == 6
    assert 'current_weather' in generator.intents
    assert 'weather_forecast' in generator.intents
    assert 'temperature_query' in generator.intents
    assert 'precipitation_query' in generator.intents
    assert 'weather_comparison' in generator.intents
    assert 'general_greeting' in generator.intents

def test_pattern_generation(generator):
    """Test if patterns are generated correctly for each intent"""
    df = generator.generate_training_data(samples_per_intent=10)
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert 'text' in df.columns
    assert 'intent' in df.columns
    assert 'entities' in df.columns
    
    # Check intent distribution
    intent_counts = df['intent'].value_counts()
    assert len(intent_counts) == 6  # All intents should be present
    
    # Check entity structure
    for entities in df['entities']:
        assert isinstance(entities, dict)

def test_location_handling(generator):
    """Test if locations are handled correctly in patterns"""
    df = generator.generate_training_data(samples_per_intent=5)
    
    # Check weather comparison patterns
    comparison_samples = df[df['intent'] == 'weather_comparison']
    for _, row in comparison_samples.iterrows():
        assert 'location1' in row['entities']
        assert 'location2' in row['entities']
        assert row['entities']['location1'] != row['entities']['location2']
    
    # Check single location patterns
    single_location_intents = ['current_weather', 'weather_forecast', 'temperature_query']
    for intent in single_location_intents:
        samples = df[df['intent'] == intent]
        for _, row in samples.iterrows():
            assert 'location' in row['entities']

def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization"""
    assert preprocessor.max_length == 50
    assert preprocessor.pad_token == '<PAD>'
    assert preprocessor.unk_token == '<UNK>'
    assert len(preprocessor.vocab) == 0  # Should be empty initially

def test_vocabulary_building(preprocessor):
    """Test vocabulary building from texts"""
    texts = [
        "What's the weather like in New York?",
        "How's the temperature in London?",
        "Will it rain in Tokyo tomorrow?"
    ]
    
    preprocessor.build_vocab(texts, min_freq=1)
    
    # Check vocabulary
    assert len(preprocessor.vocab) > 0
    assert preprocessor.pad_token in preprocessor.vocab
    assert preprocessor.unk_token in preprocessor.vocab
    
    # Check word mappings
    assert all(word in preprocessor.word_to_idx for word in preprocessor.vocab)
    assert all(idx in preprocessor.idx_to_word for idx in range(len(preprocessor.vocab)))

def test_text_to_sequence(preprocessor):
    """Test text to sequence conversion"""
    # Build vocabulary first
    texts = ["What's the weather like in New York?"]
    preprocessor.build_vocab(texts, min_freq=1)
    
    # Test sequence conversion
    sequence = preprocessor.text_to_sequence("What's the weather like in London?")
    
    # Check sequence properties
    assert len(sequence) == preprocessor.max_length
    assert all(isinstance(x, int) for x in sequence)
    
    # Test unknown word handling
    sequence = preprocessor.text_to_sequence("completely new words here")
    assert all(idx == preprocessor.word_to_idx[preprocessor.unk_token] for idx in sequence)

def test_preprocessor_save_load(tmp_path, preprocessor):
    """Test saving and loading preprocessor state"""
    # Build vocabulary
    texts = ["What's the weather like in New York?"]
    preprocessor.build_vocab(texts, min_freq=1)
    
    # Save preprocessor
    save_path = tmp_path / "preprocessor.json"
    preprocessor.save_preprocessor(save_path)
    
    # Load preprocessor
    loaded_preprocessor = TextPreprocessor.load_preprocessor(save_path)
    
    # Check if state is preserved
    assert loaded_preprocessor.max_length == preprocessor.max_length
    assert loaded_preprocessor.vocab == preprocessor.vocab
    assert loaded_preprocessor.word_to_idx == preprocessor.word_to_idx
    assert loaded_preprocessor.idx_to_word == preprocessor.idx_to_word

def test_dataset_saving(generator, tmp_path):
    """Test dataset saving functionality"""
    df = generator.generate_training_data(samples_per_intent=10)
    generator.save_dataset(df, output_dir=tmp_path)
    
    # Check if files are created
    assert (tmp_path / 'weather_dataset.csv').exists()
    assert (tmp_path / 'train.csv').exists()
    assert (tmp_path / 'test.csv').exists()
    assert (tmp_path / 'intent_mapping.json').exists()
    
    # Check if train/test split is correct
    train_df = pd.read_csv(tmp_path / 'train.csv')
    test_df = pd.read_csv(tmp_path / 'test.csv')
    
    assert len(train_df) + len(test_df) == len(df)
    assert len(train_df) > len(test_df)  # Train set should be larger 