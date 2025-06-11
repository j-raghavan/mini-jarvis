import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDatasetGenerator:
    def __init__(self):
        self.intents = {
            'current_weather': 0,
            'weather_forecast': 1, 
            'temperature_query': 2,
            'precipitation_query': 3,
            'weather_comparison': 4,
            'general_greeting': 5
        }
        
        # Define patterns for each intent
        self.patterns = {
            'current_weather': [
                "What's the weather like in {location}?",
                "How's the weather today in {location}?",
                "Current weather conditions for {location}",
                "Tell me about today's weather in {location}",
                "What's the weather right now in {location}?",
                "Can you tell me the weather in {location}?",
                "Weather update for {location}",
                "What's it like outside in {location}?",
                "Is it raining in {location}?",
                "What's the temperature in {location}?",
                "Weather report for {location}",
                "How's the weather looking in {location}?",
                "What's the current weather in {location}?",
                "Weather conditions in {location}",
                "What's the weather situation in {location}?"
            ],
            'weather_forecast': [
                "What's the weather forecast for {location}?",
                "Will it rain tomorrow in {location}?",
                "Weather forecast for {location} for the next few days",
                "What's the weather going to be like in {location} tomorrow?",
                "Tell me the weather forecast for {location}",
                "What's the weather prediction for {location}?",
                "Will it be sunny in {location} tomorrow?",
                "Weather outlook for {location}",
                "What's the weather going to be like this week in {location}?",
                "Forecast for {location}",
                "Weather prediction for {location}",
                "What's the weather going to be like in {location} this weekend?",
                "Will it snow in {location} tomorrow?",
                "Weather forecast for the week in {location}",
                "What's the weather going to be like in {location} next week?"
            ],
            'temperature_query': [
                "What's the temperature in {location}?",
                "How hot is it in {location}?",
                "What's the current temperature in {location}?",
                "Temperature reading for {location}",
                "How cold is it in {location}?",
                "What's the temp in {location}?",
                "Current temperature in {location}",
                "What's the mercury reading in {location}?",
                "How warm is it in {location}?",
                "Temperature in {location} right now",
                "What's the heat index in {location}?",
                "How's the temperature in {location}?",
                "What's the feels like temperature in {location}?",
                "Temperature conditions in {location}",
                "What's the temperature going to be in {location}?"
            ],
            'precipitation_query': [
                "Is it going to rain in {location}?",
                "What's the chance of rain in {location}?",
                "Will it rain today in {location}?",
                "Rain forecast for {location}",
                "What's the precipitation chance in {location}?",
                "Is there any rain expected in {location}?",
                "Will it snow in {location}?",
                "What's the chance of snow in {location}?",
                "Precipitation forecast for {location}",
                "Rain probability in {location}",
                "Will there be any precipitation in {location}?",
                "What's the rain forecast for {location}?",
                "Is there any chance of rain in {location}?",
                "Will it drizzle in {location}?",
                "What's the snow forecast for {location}?"
            ],
            'weather_comparison': [
                "Compare weather in {location1} and {location2}",
                "What's the weather difference between {location1} and {location2}?",
                "How does the weather in {location1} compare to {location2}?",
                "Weather comparison between {location1} and {location2}",
                "Which is warmer, {location1} or {location2}?",
                "Compare temperatures in {location1} and {location2}",
                "Weather contrast between {location1} and {location2}",
                "How's the weather different in {location1} vs {location2}?",
                "Temperature comparison between {location1} and {location2}",
                "Which city has better weather, {location1} or {location2}?",
                "Compare current conditions in {location1} and {location2}",
                "Weather difference between {location1} and {location2}",
                "How do temperatures compare in {location1} and {location2}?",
                "Which is colder, {location1} or {location2}?",
                "Weather comparison: {location1} versus {location2}"
            ],
            'general_greeting': [
                "Hello",
                "Hi there",
                "Hey",
                "Good morning",
                "Good afternoon",
                "Good evening",
                "How are you?",
                "What's up?",
                "Greetings",
                "Hello there",
                "Hi, how are you?",
                "Hey there",
                "Good day",
                "Hello, how can you help me?",
                "Hi, I need some weather information"
            ]
        }
        
        # Common locations for data generation
        self.locations = [
            "New York", "San Francisco", "London", "Tokyo", "Paris",
            "Sydney", "Berlin", "Moscow", "Dubai", "Singapore",
            "Toronto", "Mumbai", "Seoul", "Rome", "Amsterdam",
            "here", "my location", "current location"
        ]
        
    def generate_training_data(self, samples_per_intent=1000):
        """Generate training data with specified number of samples per intent"""
        samples = []
        
        # Generate samples for each intent
        for intent, patterns in self.patterns.items():
            if intent == 'weather_comparison':
                # Special handling for comparison intent
                for pattern in patterns:
                    for loc1 in self.locations:
                        for loc2 in self.locations:
                            if loc1 != loc2:
                                samples.append({
                                    'text': pattern.format(location1=loc1, location2=loc2),
                                    'intent': intent,
                                    'entities': {'location1': loc1, 'location2': loc2}
                                })
            elif intent == 'general_greeting':
                # No location needed for greetings
                for pattern in patterns:
                    samples.append({
                        'text': pattern,
                        'intent': intent,
                        'entities': {}
                    })
            else:
                # For all other intents
                for pattern in patterns:
                    for location in self.locations:
                        samples.append({
                            'text': pattern.format(location=location),
                            'intent': intent,
                            'entities': {'location': location}
                        })
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} total samples")
        logger.info(f"Samples per intent: {df['intent'].value_counts().to_dict()}")
        
        return df
    
    def save_dataset(self, df, output_dir='data'):
        """Save the generated dataset to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save the full dataset
        df.to_csv(output_path / 'weather_dataset.csv', index=False)
        
        # Save intent mapping
        with open(output_path / 'intent_mapping.json', 'w') as f:
            json.dump(self.intents, f, indent=2)
        
        # Split and save train/test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['intent'])
        
        train_df.to_csv(output_path / 'train.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        logger.info(f"Saved dataset to {output_path}")
        logger.info(f"Train set size: {len(train_df)}")
        logger.info(f"Test set size: {len(test_df)}")

class TextPreprocessor:
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from training texts"""
        # Count word frequencies
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter by minimum frequency
        vocab = {word for word, freq in word_freq.items() if freq >= min_freq}
        
        # Add special tokens
        vocab.add(self.pad_token)
        vocab.add(self.unk_token)
        
        # Create mappings
        self.vocab = sorted(list(vocab))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")
        
    def text_to_sequence(self, text):
        """Convert text to sequence of token indices"""
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, self.word_to_idx[self.unk_token]) for word in words]
        
        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            sequence.extend([self.word_to_idx[self.pad_token]] * (self.max_length - len(sequence)))
        else:
            sequence = sequence[:self.max_length]
            
        return sequence
    
    def save_preprocessor(self, output_path):
        """Save preprocessor state"""
        state = {
            'max_length': self.max_length,
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved preprocessor state to {output_path}")
    
    @classmethod
    def load_preprocessor(cls, input_path):
        """Load preprocessor state"""
        with open(input_path, 'r') as f:
            state = json.load(f)
        
        preprocessor = cls(max_length=state['max_length'])
        preprocessor.vocab = state['vocab']
        preprocessor.word_to_idx = state['word_to_idx']
        preprocessor.idx_to_word = state['idx_to_word']
        
        logger.info(f"Loaded preprocessor state from {input_path}")
        return preprocessor

if __name__ == "__main__":
    # Generate and save dataset
    generator = WeatherDatasetGenerator()
    df = generator.generate_training_data()
    generator.save_dataset(df)
    
    # Build and save preprocessor
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(df['text'])
    preprocessor.save_preprocessor('data/preprocessor.json') 