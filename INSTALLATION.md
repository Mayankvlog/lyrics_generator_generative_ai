# Installation Guide

Comprehensive installation and setup instructions for the Lyrics Generator project.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Local Development Setup](#local-development-setup)
- [Docker Setup](#docker-setup)
- [Database Configuration](#database-configuration)
- [Model Files](#model-files)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|------------|
| Python | 3.11 or higher |
| RAM | 4 GB minimum (8 GB recommended) |
| Disk Space | 2 GB for dependencies + models |
| GPU | Optional (NVIDIA GPU for faster inference) |
| OS | Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+) |

### Optional Requirements
- **MongoDB**: For database-backed lyrics storage
- **Docker**: For containerized deployment
- **GPU Support**: NVIDIA GPU with CUDA 12.0+ for faster inference
- **Git**: For cloning the repository

## Installation Methods

### Method 1: Local Development (Recommended for Development)

#### Windows PowerShell

```powershell
# 1. Clone repository
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
cd lyrics_generator

# 2. Create virtual environment
python -m venv myenv

# 3. Activate virtual environment
myenv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import streamlit; print('Streamlit OK')"
```

#### macOS/Linux

```bash
# 1. Clone repository
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
cd lyrics_generator

# 2. Create virtual environment
python3 -m venv myenv

# 3. Activate virtual environment
source myenv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import streamlit; print('Streamlit OK')"
```

### Method 2: Docker (Recommended for Production)

```bash
# 1. Clone repository
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
cd lyrics_generator

# 2. Build image
docker build -t lyrics-generator:latest .

# 3. Run container
docker run -p 8502:8502 \
  -e MONGODB_URI="your_mongodb_uri" \
  lyrics-generator:latest

# 4. Access at http://localhost:8502
```

### Method 3: Docker Compose (Easiest for Production)

```bash
# 1. Clone repository
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
cd lyrics_generator

# 2. Create .env file
cp .env.example .env
# Edit .env with your configuration

# 3. Start services
docker-compose up -d

# 4. View logs
docker-compose logs -f app

# 5. Access at http://localhost:8502

# 6. Stop services (when done)
docker-compose down
```

## Local Development Setup

### Step 1: Clone and Prepare

```bash
git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
cd lyrics_generator
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# (Optional) Install development tools
pip install pytest pytest-cov black flake8 mypy jupyter notebook
```

### Step 4: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (use your favorite editor)
# Windows
notepad .env

# macOS/Linux
nano .env
```

**Required variables in .env:**
```env
MONGODB_URI=your_mongodb_connection_string
DOCKERHUB_USERNAME=your_dockerhub_username
```

### Step 5: Verify Installation

```bash
# Test Python imports
python -c "import tensorflow, streamlit, sklearn; print('All packages installed!')"

# Start the application
python -m streamlit run main.py
```

Application should open at: **http://localhost:8501**

## Docker Setup

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+ (for compose method)

### Build Docker Image

```bash
# Build with tag
docker build -t lyrics-generator:latest .

# Build with custom tag
docker build -t your-username/lyrics-generator:v1.0 .

# View built images
docker images | grep lyrics
```

### Run Docker Container

```bash
# Basic run
docker run -p 8502:8502 lyrics-generator:latest

# With environment variables
docker run -p 8502:8502 \
  -e MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/db" \
  lyrics-generator:latest

# With volume mount for development
docker run -p 8502:8502 \
  -v $(pwd):/app \
  lyrics-generator:latest

# Run in background
docker run -d -p 8502:8502 \
  --name lyrics-app \
  lyrics-generator:latest

# View logs
docker logs -f lyrics-app
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View specific logs
docker-compose logs -f app

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Database Configuration

### MongoDB Setup (Optional)

#### Using MongoDB Atlas (Cloud)

1. **Create MongoDB Atlas Account**
   - Go to https://www.mongodb.com/cloud/atlas
   - Sign up and create a project

2. **Create Cluster**
   - Create a free cluster
   - Configure security rules
   - Whitelist your IP address

3. **Get Connection String**
   - Go to "Database" → "Connect"
   - Select "Connect with your application"
   - Copy the connection string

4. **Configure in .env**
   ```env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/food?retryWrites=true&w=majority
   ```

5. **Test Connection**
   ```python
   from pymongo import MongoClient
   client = MongoClient("mongodb+srv://...")
   print("Connected!" if client else "Failed")
   ```

#### Using Local MongoDB

1. **Install MongoDB Community Edition**
   - Windows: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/
   - macOS: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-macos/
   - Linux: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-linux/

2. **Start MongoDB Service**
   ```bash
   # Windows
   mongod --dbpath "C:\data\db"
   
   # macOS/Linux
   brew services start mongodb-community
   ```

3. **Configure in .env**
   ```env
   MONGODB_URI=mongodb://localhost:27017/food
   ```

### CSV Data Setup

1. **Prepare CSV File**
   - Ensure you have a CSV file with a `Lyric` column
   - Example: `ArianaGrande.csv`

2. **Place in Project Root**
   ```
   lyrics_generator/
   ├── ArianaGrande.csv
   ├── main.py
   └── ...
   ```

3. **Configure in main.py**
   ```python
   csv_df = pd.read_csv('ArianaGrande.csv')
   # Ensure 'Lyric' column exists
   ```

## Model Files

### Required Model Files

The application requires three pre-trained model files:

```
models/
├── rag_lyrics_model.h5          # Keras model weights
├── tokenizer.pickle             # Word tokenizer
└── tfidf_vectorizer.pkl         # TF-IDF vectorizer
```

### Obtaining Model Files

1. **From GitHub Releases**
   ```bash
   # Download from releases page
   # https://github.com/Mayankvlog/lyrics_generator_generative_ai/releases
   ```

2. **From Provided Package**
   - Models are included in the repository
   - Located in `models/` directory

3. **Training Your Own Models**
   - See `lyrics_generator.ipynb` notebook
   - Follow the training workflow
   - Models will be saved to `models/` directory

### Verifying Model Files

```bash
# Check if all required files exist
ls -la models/

# On Windows
dir models\

# Python verification
import os
required_files = [
    'models/rag_lyrics_model.h5',
    'models/tokenizer.pickle',
    'models/tfidf_vectorizer.pkl'
]
for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING")
```

## Troubleshooting

### Python Version Issues

```bash
# Check Python version
python --version

# If not 3.11+, use specific version
python3.11 -m venv myenv
```

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Check for conflicts
pip check

# Upgrade pip
pip install --upgrade pip
```

### Port Already in Use

```bash
# Windows - Find and kill process on port 8501
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8501
kill -9 <PID>

# Use different port
streamlit run main.py --server.port 8503
```

### MongoDB Connection Issues

```bash
# Test connection
python -c "from pymongo import MongoClient; MongoClient('your_uri')"

# Check MongoDB URI format
# mongodb://host:port/database
# mongodb+srv://username:password@cluster.mongodb.net/database

# Whitelist IP in MongoDB Atlas security rules
```

### Model File Issues

```bash
# Download models manually from GitHub
git clone --sparse https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
git sparse-checkout set models
```

### Memory Issues

```bash
# Reduce batch size in main.py
# Allocate more RAM to Docker container
docker run -m 4g lyrics-generator:latest
```

### GPU Support (Optional)

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Verification

### Test Installation

```python
# test_installation.py
import sys
import importlib

packages = [
    'streamlit',
    'tensorflow',
    'keras',
    'pandas',
    'numpy',
    'sklearn',
    'pymongo',
    'h5py',
    'joblib'
]

print(f"Python {sys.version}")
print("-" * 50)

for package in packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'installed')
        print(f"✓ {package}: {version}")
    except ImportError:
        print(f"✗ {package}: NOT INSTALLED")

print("-" * 50)
print("Installation verification complete!")
```

Run verification:
```bash
python test_installation.py
```

### Launch Application

```bash
# Local
python -m streamlit run main.py

# Docker
docker run -p 8502:8502 lyrics-generator:latest

# Docker Compose
docker-compose up
```

### Access Application

- Local: **http://localhost:8501**
- Docker: **http://localhost:8502**
- Remote: **http://your-server-ip:8502**

## Getting Help

If you encounter issues:

1. **Check documentation**: [README.md](README.md)
2. **Search issues**: [GitHub Issues](https://github.com/Mayankvlog/lyrics_generator_generative_ai/issues)
3. **Open new issue**: Include Python version, OS, error message
4. **Email**: mayankkr0311@gmail.com

## Next Steps

- ✅ Installation complete
- 📖 Read [README.md](README.md) for project overview
- 🚀 Launch the application
- 📝 Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- 🚢 See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
