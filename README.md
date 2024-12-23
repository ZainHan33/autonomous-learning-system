# Autonomous Learning System

A Python-based autonomous learning system implementing a split-system architecture based on cognitive science principles.

## Features
- Split system architecture (surface and deep processing)
- Autonomous web learning using Brave Search API
- NLP-based content processing
- Pattern recognition and concept mapping
- Interactive visualization of learned concepts

## Installation

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate learner

# Install spaCy English model
python -m spacy download en_core_web_sm
```

## Configuration
Add your Brave Search API key to `config.py`:
```python
BRAVE_API_KEY = "your-api-key-here"
```

## Usage
```python
python example.py
```

## Project Structure
- `models/` - Neural network components
  - `surface_system.py` - Surface processing system
  - `deep_system.py` - Deep processing system
- `content_processor.py` - NLP-based content processing
- `search_manager.py` - Web search functionality
- `visualizer.py` - Concept visualization
- `learning_manager.py` - Main learning system
- `example.py` - Usage example