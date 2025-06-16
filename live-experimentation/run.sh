#!/bin/bash

echo "🧠 Starting Live Token Generation Experiment..."
echo "========================================"

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Make sure you're in the live-experimentation directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Start the application
echo "🚀 Starting Flask application..."
echo "📖 Open http://localhost:5000 in your browser"
echo "🛑 Press Ctrl+C to stop the server"
echo "========================================"

python app.py 