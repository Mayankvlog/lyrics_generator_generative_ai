# lyrics_generator_generative_ai


#   🎵 AI Lyrics Generator - RAG-Based Generative Model

An advanced lyrics generation system combining Retrieval-Augmented Generation (RAG) with deep learning RNN architecture. The model uses 5 LSTM layers with diverse activation functions to create contextually relevant and creative song lyrics based on user prompts.


🚀 Project Overview

This generative AI system integrates retrieval and generation mechanisms to produce original lyrics. The model retrieves relevant lyrical patterns using TF-IDF vectorization and cosine similarity, then generates new lyrics using a trained 5-layer LSTM neural network with temperature sampling for controlled creativity.
✨ Key Features

    5-Layer LSTM Architecture: Deep RNN with tanh, ELU, SELU, and sigmoid activation functions

    RAG Implementation: Combines retrieval and generation for context-aware lyrics

    MongoDB Integration: Connects to cloud database for scalable lyric storage

    Temperature Sampling: Adjustable creativity parameter for generation control

    Interactive Streamlit UI: User-friendly interface with real-time generation

    TF-IDF Retrieval: Semantic search using scikit-learn vectorization

    Robust Error Handling: Gracefully manages unknown words and edge cases

🛠️ Technology Stack

Deep Learning: TensorFlow, Keras (LSTM layers)
Database: MongoDB Atlas (cloud-hosted)
Retrieval: TF-IDF, cosine similarity (scikit-learn)
Frontend: Streamlit
Data Processing: pandas, NumPy
Serialization: pickle, joblib
📦 Installation

bash
# Clone repository
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git

cd lyrics_generator_generative_ai

# Install dependencies
pip install tensorflow pandas numpy scikit-learn pymongo streamlit joblib

# Run Streamlit app
streamlit run main.py

🎯 Usage

    Launch the application

    Enter a theme or prompt (e.g., "I'm feeling sad")

    Adjust creativity slider (temperature: 0.1-1.5)

    Set number of words to generate (20-5000)

    Click "Generate Lyrics"

    View retrieved context and generated output

#  🏗️ Model Architecture

Input Layer: Embedding (vocab_size × 128)
Hidden Layers: 5 LSTM layers (256 units each)
Dropout: 0.2 between layers
Output Layer: Dense (softmax activation)
Loss Function: Categorical crossentropy
Optimizer: Adam with gradient clipping
📊 Project Structure

text
├── lyrics_generator.ipynb    # Training notebook

├── main.py                    # Streamlit application

├── models/

│   ├── rag_lyrics_model.h5   # Trained model

│   ├── tokenizer.pickle       # Keras tokenizer

│   └── tfidf_vectorizer.pkl   # TF-IDF vectorizer

└── ArianaGrande.csv          # Dataset



