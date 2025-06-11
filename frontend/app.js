class WeatherAssistant {
    constructor() {
        console.log('Initializing WeatherAssistant...');
        try {
            // API configuration - hardcoded for local development
            this.apiUrl = 'http://localhost:8000';
            console.log('API URL set to:', this.apiUrl);

            this.isRecording = false;
            this.autoSpeak = true;

            // Speech APIs
            this.recognition = null;
            this.synthesis = window.speechSynthesis;

            // Initialize elements first
            this.initializeElements();
            console.log('Elements initialized');

            // Verify all required elements are present
            const missingElements = [];
            if (!this.chatContainer) missingElements.push('chatContainer');
            if (!this.messageInput) missingElements.push('messageInput');
            if (!this.sendButton) missingElements.push('sendButton');
            if (!this.voiceButton) missingElements.push('voiceButton');
            if (!this.clearButton) missingElements.push('clearButton');
            if (!this.autoSpeakCheckbox) missingElements.push('autoSpeak');
            if (!this.statusText) missingElements.push('statusText');

            if (missingElements.length > 0) {
                throw new Error(`Required elements not found in the DOM: ${missingElements.join(', ')}`);
            }

            // Set up event listeners first
            this.setupEventListeners();
            console.log('Event listeners set up');

            // Then initialize speech recognition
            this.initializeSpeechRecognition();
            console.log('Speech recognition initialized');

            console.log('WeatherAssistant initialized successfully');
            // Initial greeting
            this.addMessage('assistant', 'Hello! I\'m your PyTorch-powered weather assistant. Ask me about the weather anywhere!');
        } catch (error) {
            console.error('Failed to initialize WeatherAssistant:', error);
            throw error; // Re-throw to be caught by the DOMContentLoaded handler
        }
    }

    initializeElements() {
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.voiceButton = document.getElementById('voiceButton');
        this.clearButton = document.getElementById('clearButton');
        this.autoSpeakCheckbox = document.getElementById('autoSpeak');
        this.statusText = document.getElementById('statusText');
    }

    initializeSpeechRecognition() {
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onstart = () => {
                this.isRecording = true;
                this.voiceButton.classList.add('recording');
                this.statusText.textContent = 'Listening...';
            };

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.messageInput.value = transcript;
                // Use the bound method if it exists, otherwise bind it
                if (this.boundSendMessage) {
                    this.boundSendMessage();
                } else {
                    this.sendMessage.bind(this)();
                }
            };

            this.recognition.onend = () => {
                this.isRecording = false;
                this.voiceButton.classList.remove('recording');
                this.statusText.textContent = 'Ready';
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.statusText.textContent = 'Speech recognition error';
                this.isRecording = false;
                this.voiceButton.classList.remove('recording');
            };
        } else {
            this.voiceButton.style.display = 'none';
            console.warn('Speech recognition not supported');
        }
    }

    setupEventListeners() {
        // Bind methods to this instance
        this.boundSendMessage = this.sendMessage.bind(this);
        this.boundToggleVoiceRecording = this.toggleVoiceRecording.bind(this);
        this.boundClearChat = this.clearChat.bind(this);

        // Send message on button click
        this.sendButton.addEventListener('click', this.boundSendMessage);

        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.boundSendMessage();
            }
        });

        // Voice recording toggle
        this.voiceButton.addEventListener('click', this.boundToggleVoiceRecording);

        // Clear chat
        this.clearButton.addEventListener('click', this.boundClearChat);

        // Auto-speak toggle
        this.autoSpeakCheckbox.addEventListener('change', (e) => {
            this.autoSpeak = e.target.checked;
        });
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        console.log('Sending message:', message);
        console.log('API endpoint:', `${this.apiUrl}/chat`);

        // Add user message to chat
        this.addMessage('user', message);
        this.messageInput.value = '';

        // Show typing indicator
        this.showTypingIndicator();
        this.statusText.textContent = 'Processing with PyTorch model...';

        try {
            console.log('Making API request to:', `${this.apiUrl}/chat`);
            // Send to backend API
            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: this.getUserId(),
                    include_audio: this.autoSpeak
                })
            });

            console.log('API Response status:', response.status);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('API Response data:', data);

            // Remove typing indicator
            this.hideTypingIndicator();

            // Add assistant response
            this.addMessage('assistant', data.text, {
                intent: data.intent,
                confidence: data.confidence,
                weatherData: data.weather_data
            });

            // Speak response if auto-speak is enabled
            if (this.autoSpeak) {
                this.speak(data.text);
            }

            this.statusText.textContent = `Intent: ${data.intent} (${(data.confidence * 100).toFixed(1)}% confidence)`;

        } catch (error) {
            console.error('Error details:', error);
            console.error('Error stack:', error.stack);
            this.hideTypingIndicator();
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            this.statusText.textContent = 'Error occurred';
        }
    }

    addMessage(sender, text, metadata = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.textContent = text;

        // Add metadata display for assistant messages
        if (sender === 'assistant' && metadata) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-metadata';
            metaDiv.innerHTML = `
                <small>
                    Intent: ${metadata.intent} | 
                    Confidence: ${(metadata.confidence * 100).toFixed(1)}%
                </small>
            `;
            bubbleDiv.appendChild(metaDiv);

            // Add weather card if weather data is available
            if (metadata.weatherData) {
                const weatherCard = document.createElement('div');
                weatherCard.className = 'weather-card';
                weatherCard.innerHTML = `
                    <div class="weather-info">
                        <div class="weather-main">
                            <span class="temperature">${Math.round(metadata.weatherData.main.temp)}°F</span>
                            <span class="condition">${metadata.weatherData.weather[0].description}</span>
                        </div>
                        <div class="weather-details">
                            <div>Feels like: ${Math.round(metadata.weatherData.main.feels_like)}°F</div>
                            <div>Humidity: ${metadata.weatherData.main.humidity}%</div>
                            <div>Wind: ${metadata.weatherData.wind.speed} mph</div>
                        </div>
                    </div>
                `;
                bubbleDiv.appendChild(weatherCard);
            }
        }

        messageDiv.appendChild(bubbleDiv);
        this.chatContainer.appendChild(messageDiv);

        // Scroll to bottom
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-indicator';
        typingDiv.id = 'typingIndicator';

        typingDiv.innerHTML = `
            <div class="message-bubble">
                <span>PyTorch model is thinking</span>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;

        this.chatContainer.appendChild(typingDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    toggleVoiceRecording() {
        if (!this.recognition) {
            alert('Speech recognition is not supported in your browser');
            return;
        }

        if (this.isRecording) {
            this.recognition.stop();
        } else {
            this.recognition.start();
        }
    }

    speak(text) {
        // Cancel any ongoing speech
        this.synthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 0.8;

        // Use a pleasant voice if available
        const voices = this.synthesis.getVoices();
        const femaleVoice = voices.find(voice =>
            voice.name.toLowerCase().includes('female') ||
            voice.name.toLowerCase().includes('zira') ||
            voice.name.toLowerCase().includes('samantha')
        );

        if (femaleVoice) {
            utterance.voice = femaleVoice;
        }

        this.synthesis.speak(utterance);
    }

    clearChat() {
        this.chatContainer.innerHTML = '';
        this.addMessage('assistant', 'Chat cleared! How can I help you with the weather?');
        this.statusText.textContent = 'Ready';
    }

    getUserId() {
        // Simple user ID generation for demo
        let userId = localStorage.getItem('weatherAssistantUserId');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('weatherAssistantUserId', userId);
        }
        return userId;
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded - Initializing WeatherAssistant...');
    try {
        window.weatherAssistant = new WeatherAssistant();
        console.log('WeatherAssistant instance created successfully');
    } catch (error) {
        console.error('Error initializing WeatherAssistant:', error);
    }
}); 