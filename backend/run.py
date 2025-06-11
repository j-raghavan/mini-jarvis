import uvicorn
import os
import socket
from dotenv import load_dotenv

def find_available_port(start_port=8000, max_port=8999):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Create static directory for audio files
    os.makedirs("static/audio", exist_ok=True)
    
    # Find an available port
    port = find_available_port()
    print(f"Starting server on port {port}")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    ) 