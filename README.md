# 🎵 Lyrics Generator - Generative AI Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange.svg)](https://www.tensorflow.org/)

An intelligent AI-powered lyrics generator built with **Retrieval-Augmented Generation (RAG)** and **TensorFlow Keras**. The application combines semantic search with neural language models to generate contextually relevant lyrics based on user input.

## 🌟 Features

- **Retrieval-Augmented Generation (RAG)**: Uses TF-IDF vectorization and cosine similarity to retrieve contextually relevant lyric snippets
- **Advanced Language Model**: TensorFlow Keras-based next-word prediction with temperature sampling for creative variation
- **Interactive Web UI**: Built with Streamlit for easy, real-time lyrics generation
- **Multi-Source Data Support**: Loads lyrics from both CSV files and MongoDB databases
- **Docker Support**: Production-ready Docker containerization for easy deployment
- **Temperature Control**: Adjust creativity level of generated lyrics (0.0 = deterministic, 1.0+ = creative)
- **Sequence Padding**: Intelligent padding for variable-length input sequences

## 🏗️ Architecture

The project combines two key AI components:

1. **Retrieval Module (RAG)**
   - TF-IDF Vectorizer for text representation
   - Cosine similarity search for context retrieval
   - Returns most semantically relevant lyric from dataset

2. **Generation Module**
   - Keras Sequential Model trained on lyric sequences
   - Word tokenization and padding
   - Temperature-based sampling for output diversity
   - Configurable sequence length (default: 100 tokens)

## 📁 Project Structure

```
lyrics_generator/
├── main.py                      # Streamlit application (main entry point)
├── lyrics_generator.ipynb       # Jupyter notebook for model training & experimentation
├── requirements.txt             # Python dependencies (71 packages)
├── Dockerfile                   # Docker containerization
├── docker-compose.yml           # Docker Compose configuration
├── .env.example                 # Environment variables template
├── ArianaGrande.csv            # Sample dataset (Ariana Grande lyrics)
├── README.md                    # This file
│
├── models/                      # Pre-trained model artifacts
│   ├── rag_lyrics_model.h5      # Trained Keras model weights
│   ├── tokenizer.pickle         # Word tokenizer for text preprocessing
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
│
├── myenv/                       # Virtual environment (local development)
│   ├── Scripts/                 # Python executables
│   ├── Lib/                     # Installed packages
│   └── pyvenv.cfg               # Virtual env configuration
│
└── .github/                     # GitHub workflows & templates
```

## 🔧 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | Python | 3.11+ |
| **ML Framework** | TensorFlow/Keras | 3.11.3 |
| **Web Framework** | Streamlit | 1.28+ |
| **Data Processing** | Pandas | 2.3.3 |
| **ML Utilities** | scikit-learn | 1.3+ |
| **Database** | MongoDB | 4.15.3 |
| **Containerization** | Docker | Latest |
| **NLP** | NumPy, SciPy | Latest |

## 📋 Prerequisites

- **Python**: 3.11 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **Docker** (optional): For containerized deployment
- **MongoDB** (optional): For database-backed lyrics storage

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
cd lyrics_generator
```

### 2. Create Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

**Local Development:**
```bash
python -m streamlit run main.py
```

Access the app at: **http://localhost:8501**

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop the service
docker-compose down
```

### Using Docker CLI

```bash
# Build the image
docker build -t lyrics-generator:latest .

# Run the container
docker run -p 8502:8502 \
  -e MONGODB_URI="your_mongodb_uri" \
  lyrics-generator:latest
```

Access the app at: **http://localhost:8502**

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (use `.env.example` as a template):

```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
VPS_HOST=your_host
VPS_USER=your_user
VPS_PASSWORD=your_password
```

### Database Configuration

The app supports data loading from:

1. **CSV Files** (Default)
   - Place CSV files in project root
   - Ensure a `Lyric` column exists
   - Example: `ArianaGrande.csv`

2. **MongoDB** (Optional)
   - Configure `MONGODB_URI` environment variable
   - Database: `food` (default)
   - Collection: `lyrics` (default)
   - Modify database/collection names in `main.py` if different

### Model Parameters

Modify these in `main.py` if needed:

```python
max_sequence_length = 100  # Must match training sequence length
temperature = 0.8          # Adjust generation creativity (0.0-2.0)
num_words = 100           # Number of words to generate
```

## 📊 Model Details

### Training Process

The model was trained on Ariana Grande lyrics using:

- **Input**: Sequences of tokens (max 100 words)
- **Output**: Next-word predictions with softmax probabilities
- **Loss Function**: Categorical crossentropy
- **Optimizer**: Adam
- **Epochs**: Trained on sequence data

### Generation Method

1. **Preprocessing**
   - User input → lowercase conversion
   - Text cleaning (remove special characters)
   - Tokenization using trained tokenizer

2. **Retrieval**
   - TF-IDF vectorization of user input
   - Cosine similarity search against dataset
   - Return most relevant lyric snippet

3. **Generation Loop**
   - Use retrieved lyric as seed
   - Iteratively predict next word (100 iterations)
   - Apply temperature sampling for diversity
   - Append predictions to generate full lyric

## 💻 Usage

### Web Interface

1. Open http://localhost:8501
2. Enter a prompt or theme (e.g., "love", "heartbreak", "dreams")
3. Adjust generation parameters:
   - **Temperature**: 0.5 (deterministic) to 2.0 (creative)
   - **Number of Words**: 50-200 (length of output)
4. Click "Generate Lyrics"
5. View generated lyrics with retrieved context

### Example Prompts

- "love and heartbreak"
- "dancing in the moonlight"
- "wish you were here"
- "breaking free"

## 📦 Dependencies

**Key Packages:**
- `streamlit` - Web UI framework
- `tensorflow` - Deep learning
- `keras` - High-level neural networks
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML utilities
- `pymongo` - MongoDB driver
- `h5py` - HDF5 file handling
- `joblib` - Model serialization

See [requirements.txt](requirements.txt) for complete list with versions.

## 📈 Performance & Optimization

- **Model Inference**: ~100-200ms per generation (GPU: ~50-100ms)
- **TF-IDF Search**: <10ms for dataset < 10,000 lyrics
- **Memory Usage**: ~500MB for model + data
- **Optimization Tips**:
  - Use GPU for faster inference: `CUDA_VISIBLE_DEVICES=0`
  - Cache models with `@st.cache_resource`
  - Batch process multiple generations

1. **Check existing issues**: [GitHub Issues](https://github.com/Mayankvlog/lyrics_generator_generative_ai/issues)
2. **Create new issue**: Include:
   - Python version & OS
   - Error message & traceback
   - Steps to reproduce
   - Expected vs actual behavior

3. **Discussions**: Use [GitHub Discussions](https://github.com/Mayankvlog/lyrics_generator_generative_ai/discussions) for general questions

## 🚀 Future Enhancements

- [ ] Multi-artist support
- [ ] Custom model training interface
- [ ] Fine-tuning with user feedback
- [ ] Real-time lyrics quality scoring
- [ ] Export to music production software
- [ ] Advanced prompt engineering
- [ ] API endpoint for backend integration
- [ ] MLOps pipeline with CI/CD
- [ ] Model versioning & A/B testing
- [ ] Analytics & usage metrics

## 📚 Related Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [RAG Pattern](https://python.langchain.com/docs/use_cases/question_answering/vector_stores_retrieval)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Made with ❤️ by Mayank Kumar | 2025**

### **Docker Execution**

```bash
# Build the image
docker build -t lyrics-generator:latest .

# Run the container
docker run -p 8501:8501 lyrics-generator:latest
```

### **Docker Compose**

```bash
# Create .env file (copy from .env.example)
cp .env.example .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app
```

---

## 🚀 VPS Deployment

### **Access your app on VPS**
- **VPS IP:** `167.71.235.91`
- **URL:** http://167.71.235.91:8501

### **Deploy to VPS (Linux)**

1. **SSH into your VPS:**
```bash
ssh root@167.71.235.91
```

2. **Run the deployment script:**
```bash
curl -fsSL https://raw.githubusercontent.com/Mayankvlog/lyrics_generator_generative_ai/main/deploy-vps.sh | bash
```

Or manually:
```bash
apt-get update && apt-get install -y git curl
curl -fsSL https://get.docker.com | sh
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
cd lyrics_generator_generative_ai
docker-compose up -d
```

3. **Access the app:**
   - Open your browser and go to: **http://167.71.235.91:8501**

### **CI/CD Deployment**

The GitHub Actions workflow automatically:
1. ✅ Tests code and dependencies
2. ✅ Builds Docker image
3. ✅ Pushes to Docker Hub
4. ✅ Deploys to VPS via SSH

