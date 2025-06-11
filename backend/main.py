from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import torch
import json
import os
import logging
from typing import Optional, Dict, Any, List
import httpx
import pyttsx3
from pathlib import Path

# Import our custom modules
from models.weather_classifier import create_model
from data.data_generator import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Weather Voice Assistant API",
    description="PyTorch-powered weather assistant with voice capabilities",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Local development
        "http://localhost:3000",  # Alternative local port
        "https://*.github.io",    # GitHub Pages
        "https://*.railway.app",  # Railway frontend
        # Add your frontend domain here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    include_audio: bool = True

class ChatResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    weather_data: Optional[Dict[str, Any]] = None
    audio_url: Optional[str] = None

# Global application state
class AppState:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.weather_client = None
        self.tts_engine = None
        self.intent_names = [
            'current_weather', 'weather_forecast', 'temperature_query',
            'precipitation_query', 'weather_comparison', 'general_greeting'
        ]

class WeatherAPIClient:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        
    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Fetch weather data from Open-Meteo API"""
        try:
            # First, get coordinates for the location
            if location.lower() in ['here', 'current', 'my location']:
                # Use IP geolocation or default location
                location = "San Francisco"  # Default for demo
                
            # Geocoding to get coordinates
            async with httpx.AsyncClient() as client:
                # Use Open-Meteo's geocoding API
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search"
                geo_params = {
                    'name': location,
                    'count': 1,
                    'language': 'en',
                    'format': 'json'
                }
                
                geo_response = await client.get(geo_url, params=geo_params)
                geo_response.raise_for_status()
                geo_data = geo_response.json()
                
                if not geo_data.get('results'):
                    raise HTTPException(status_code=400, detail="Location not found")
                
                # Get coordinates from geocoding response
                result = geo_data['results'][0]
                latitude = result['latitude']
                longitude = result['longitude']
                location_name = result['name']
                
                # Get weather data using coordinates
                weather_url = f"{self.base_url}/forecast"
                weather_params = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m',
                    'temperature_unit': 'fahrenheit',
                    'wind_speed_unit': 'mph',
                    'precipitation_unit': 'inch',
                    'timezone': 'auto'
                }
                
                weather_response = await client.get(weather_url, params=weather_params)
                weather_response.raise_for_status()
                weather_data = weather_response.json()
                
                # Transform the data to match our expected format
                current = weather_data['current']
                return {
                    'name': location_name,
                    'weather': [{
                        'description': self._get_weather_description(current['weather_code'])
                    }],
                    'main': {
                        'temp': current['temperature_2m'],
                        'feels_like': current['temperature_2m'],  # Open-Meteo doesn't provide feels_like
                        'humidity': current['relative_humidity_2m'],
                        'pressure': 1013  # Default value as Open-Meteo doesn't provide pressure
                    },
                    'wind': {
                        'speed': current['wind_speed_10m']
                    },
                    'clouds': {
                        'all': 0  # Open-Meteo doesn't provide cloud coverage
                    }
                }
                
        except httpx.HTTPError as e:
            logger.error(f"Weather API error: {str(e)}")
            raise HTTPException(status_code=400, detail="Weather data not found")
    
    def _get_weather_description(self, code: int) -> str:
        """Convert Open-Meteo weather codes to descriptions"""
        weather_codes = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            56: "light freezing drizzle",
            57: "dense freezing drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            66: "light freezing rain",
            67: "heavy freezing rain",
            71: "slight snow fall",
            73: "moderate snow fall",
            75: "heavy snow fall",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "unknown")

class WeatherAssistant:
    def __init__(self, model, preprocessor, weather_client, tts_engine):
        self.model = model
        self.preprocessor = preprocessor
        self.weather_client = weather_client
        self.tts_engine = tts_engine
        
        # Configure TTS engine for better clarity
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to find a female voice for better clarity
            female_voice = next((v for v in voices if 'female' in v.name.lower()), voices[0])
            self.tts_engine.setProperty('voice', female_voice.id)
        self.tts_engine.setProperty('rate', 150)  # Slower rate for better clarity
        self.tts_engine.setProperty('volume', 1.0)  # Maximum volume
        
        self.response_templates = {
            'current_weather': [
                "The current weather in {location} is {temp} degrees Fahrenheit with {condition}.",
                "In {location}, it is currently {temp} degrees with {condition} conditions.",
                "Right now, {location} has a temperature of {temp} degrees Fahrenheit and the weather is {condition}."
            ],
            'weather_forecast': [
                "The forecast for {location} shows {condition} weather with a temperature of {temp} degrees Fahrenheit.",
                "Tomorrow's weather in {location} will be {condition} with temperatures reaching {temp} degrees.",
                "For {location}, expect {condition} conditions with temperatures around {temp} degrees Fahrenheit."
            ],
            'temperature_query': [
                "The current temperature in {location} is {temp} degrees Fahrenheit.",
                "In {location}, the temperature is {temp} degrees right now.",
                "{location} is currently experiencing a temperature of {temp} degrees Fahrenheit."
            ],
            'general_greeting': [
                "Hello! I am your weather assistant. How may I help you today?",
                "Hi there! I can provide weather information for any location. What would you like to know?",
                "Welcome! I am here to help you with weather information. What can I tell you about?"
            ]
        }
    
    async def process_message(self, message: str, user_id: str, include_audio: bool = True) -> ChatResponse:
        """Process user message and generate response"""
        try:
            logger.info(f"Processing message: '{message}'")
            
            # 1. Intent classification
            logger.info("Starting intent classification...")
            intent, confidence = self._classify_intent(message)
            logger.info(f"Intent classified as: {intent} (confidence: {confidence:.2f})")
            
            # 2. Entity extraction
            logger.info("Starting entity extraction...")
            entities = self._extract_entities(message)
            logger.info(f"Extracted entities: {entities}")
            
            # 3. Weather data retrieval
            weather_data = None
            if intent in ['current_weather', 'weather_forecast', 'temperature_query']:
                logger.info("Fetching weather data...")
                location = entities.get('location', 'current')
                logger.info(f"Location for weather query: {location}")
                weather_data = await self.weather_client.get_weather(location)
                logger.info("Weather data retrieved successfully")
            
            # 4. Response generation
            logger.info("Generating response...")
            response_text = self._generate_response(intent, entities, weather_data)
            logger.info(f"Generated response: {response_text}")
            
            # 5. Generate audio if requested
            audio_url = None
            if include_audio:
                logger.info("Generating audio...")
                audio_url = await self._generate_audio(response_text, user_id)
                logger.info(f"Audio generated: {audio_url}")
            
            return ChatResponse(
                text=response_text,
                intent=intent,
                confidence=confidence,
                weather_data=weather_data,
                audio_url=audio_url
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)  # Added exc_info=True for full traceback
            raise HTTPException(status_code=500, detail=str(e))  # Include actual error message
    
    def _classify_intent(self, text: str) -> tuple[str, float]:
        """Classify user intent using PyTorch model"""
        # Preprocess text
        sequence = self.preprocessor.text_to_sequence(text)
        input_tensor = torch.tensor([sequence], dtype=torch.long)
        
        # Model inference
        with torch.no_grad():
            intent_logits, _ = self.model(input_tensor)
            probabilities = torch.softmax(intent_logits, dim=1)
            
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
            
            intent = app_state.intent_names[predicted_idx]
            
        return intent, confidence
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities like location, time from text"""
        entities = {}
        
        # Simple entity extraction (can be enhanced with NER model)
        words = text.lower().split()
        
        # Location detection - first check for special cases
        special_locations = ['here', 'my location', 'current location']
        for location in special_locations:
            if location in text.lower():
                entities['location'] = location
                return entities
        
        # For other locations, we'll let the weather client handle the geocoding
        # Look for location indicators in the text
        location_indicators = ['in', 'at', 'for']
        for i, word in enumerate(words):
            if word in location_indicators and i + 1 < len(words):
                # Try to extract location phrase (could be multiple words)
                location_phrase = []
                for j in range(i + 1, len(words)):
                    if words[j] in ['weather', 'temperature', 'forecast', 'today', 'tomorrow']:
                        break
                    location_phrase.append(words[j])
                if location_phrase:
                    entities['location'] = ' '.join(location_phrase)
                    break
        
        # If no location found with indicators, try to find location-like phrases
        if 'location' not in entities:
            # Look for capitalized words or phrases that might be locations
            import re
            location_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            if location_matches:
                entities['location'] = location_matches[0]
        
        # Time detection
        time_words = ['today', 'tomorrow', 'tonight', 'this week']
        for time_word in time_words:
            if time_word in words:
                entities['time'] = time_word
                break
                
        return entities
    
    def _generate_response(self, intent: str, entities: Dict[str, str], 
                          weather_data: Optional[Dict[str, Any]]) -> str:
        """Generate natural language response"""
        if intent == 'general_greeting':
            import random
            return random.choice(self.response_templates[intent])
        
        if weather_data is None:
            return "I apologize, but I couldn't retrieve the weather information at this time. Please try again."
        
        # Extract weather information and format for clarity
        temp = round(weather_data['main']['temp'])
        condition = weather_data['weather'][0]['description'].capitalize()  # Capitalize first letter
        location = weather_data['name']
        
        # Add more context for temperature
        if temp < 32:
            temp_context = "very cold"
        elif temp < 50:
            temp_context = "cold"
        elif temp < 70:
            temp_context = "mild"
        elif temp < 85:
            temp_context = "warm"
        else:
            temp_context = "hot"
            
        # Format the condition for better readability
        condition = f"{condition} ({temp_context})"
        
        import random
        template = random.choice(self.response_templates.get(intent, 
                                                          self.response_templates['current_weather']))
        
        return template.format(
            temp=temp,
            condition=condition,
            location=location
        )
    
    async def _generate_audio(self, text: str, user_id: str) -> str:
        """Generate audio file for response"""
        try:
            # Create audio directory if it doesn't exist
            audio_dir = Path("static/audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            filename = f"response_{user_id}_{hash(text)}.mp3"
            filepath = audio_dir / filename
            
            # Generate audio file
            self.tts_engine.save_to_file(text, str(filepath))
            self.tts_engine.runAndWait()
            
            return f"/audio/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return None

# Global variables - moved after all class definitions
app_state = AppState()
weather_client = WeatherAPIClient()
weather_assistant = None  # Will be initialized during startup

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global weather_assistant  # Need to declare global to modify it
    try:
        logger.info("Loading PyTorch model...")
        
        # Load preprocessor
        preprocessor_path = Path(__file__).parent / "data" / "preprocessor.json"
        logger.info(f"Loading preprocessor from: {preprocessor_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Please train the model first.")
        
        app_state.preprocessor = TextPreprocessor.load_preprocessor(preprocessor_path)
        logger.info("Preprocessor loaded successfully")
        
        # Load model
        model_path = Path(__file__).parent / "models" / "best_model_epoch_19.pth"
        logger.info(f"Loading model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        # Create and load model
        app_state.model = create_model(
            vocab_size=len(app_state.preprocessor.vocab),
            model_type='basic'  # Using basic model for now
        )
        
        # Load checkpoint and extract model state
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load from checkpoint dictionary
            app_state.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} "
                       f"with validation accuracy: {checkpoint.get('val_accuracy', 'unknown'):.2%}")
        else:
            # Load directly as state dict
            app_state.model.load_state_dict(checkpoint)
            
        app_state.model.eval()
        
        # Initialize TTS engine
        app_state.tts_engine = pyttsx3.init()
        voices = app_state.tts_engine.getProperty('voices')
        if voices:
            app_state.tts_engine.setProperty('voice', voices[0].id)
        app_state.tts_engine.setProperty('rate', 180)
        app_state.tts_engine.setProperty('volume', 0.9)
        
        # Initialize WeatherAssistant
        weather_assistant = WeatherAssistant(
            model=app_state.model,
            preprocessor=app_state.preprocessor,
            weather_client=weather_client,
            tts_engine=app_state.tts_engine
        )
        logger.info("WeatherAssistant initialized successfully")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    if weather_assistant is None:
        raise HTTPException(status_code=503, detail="WeatherAssistant not initialized")
    return await weather_assistant.process_message(
        message=request.message,
        user_id=request.user_id,
        include_audio=request.include_audio
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": app_state.model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return {
        "model_type": "PyTorch LSTM + Attention",
        "parameters": sum(p.numel() for p in app_state.model.parameters()),
        "device": next(app_state.model.parameters()).device.type,
        "intents": app_state.intent_names
    }

@app.get("/weather/{location}")
async def get_weather(location: str):
    """Direct weather data endpoint"""
    try:
        weather_data = await weather_client.get_weather(location)
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 