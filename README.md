# 🎵 Lyrics Generator (Streamlit + RAG)

A simple, local AI lyrics generator built with:
- Retrieval-Augmented Generation (TF‑IDF + cosine similarity) for context
- A TensorFlow Keras next‑word model with temperature sampling for generation
- A Streamlit UI for interactive use

The app retrieves a relevant lyric snippet from your dataset and then generates new lyrics conditioned on that context.

---

## Project structure

```
lyrics_generator/
├── main.py                  # Streamlit app
├── lyrics_generator.ipynb   # Notebook used during experimentation
├── requirements.txt         # Python dependencies
├── ArianaGrande.csv         # Dataset 
└── models/                  # Pretrained artifacts used by the app
    ├── rag_lyrics_model.h5
    ├── tokenizer.pickle
    └── tfidf_vectorizer.pkl
```

> Note: Checkpoints under `.ipynb_checkpoints/` are auto-created by Jupyter and can be ignored.

---

## Prerequisites
- Python 3.10+ recommended
- pip
- (Optional) Access to a MongoDB instance if you want to augment data from a database

---

## Setup

Create and activate a virtual environment, then install dependencies.

- Windows PowerShell
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- macOS/Linux
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## Configuration (very important)

The app can read lyrics from both a CSV and MongoDB. Do NOT hardcode secrets in code. Set an environment variable for your MongoDB connection string and use that in `main.py`.

- Windows PowerShell
  ```powershell
  $env:MONGODB_URI="{{MONGODB_URI}}"
  ```

- macOS/Linux
  ```bash
  export MONGODB_URI="{{MONGODB_URI}}"
  ```

Defaults in code expect database `food` and collection `lyrics`. Adjust if your setup differs.

Also ensure the following files exist:
- `models/rag_lyrics_model.h5`
- `models/tokenizer.pickle`
- `models/tfidf_vectorizer.pkl`
- `ArianaGrande.csv` with a `Lyric` column

> Tip: If you trained with a different sequence length, update `max_sequence_length` in `main.py` to match training.

---

## Run the app

### **Local Execution (Windows)**

```bash
# Activate virtual environment
myenv\Scripts\Activate.ps1

# Run Streamlit app
python -m streamlit run main.py
```

Or use the provided batch script:
```bash
run-app.bat
```

Access the app at: **http://localhost:8501**

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

Add these secrets to your GitHub repository:
- `DOCKERHUB_USERNAME` - Your Docker Hub username
- `DOCKERHUB_TOKEN` - Your Docker Hub access token
- `VPS_HOST` - VPS IP address (167.71.235.91)
- `VPS_USER` - SSH username (root)
- `VPS_PASSWORD` - SSH password

---

## Troubleshooting
- “Error loading artifacts” → Ensure all files are present under `models/` with the exact names shown above.
- “ArianaGrande.csv not found” → Place it next to `main.py`, confirm it has a `Lyric` column.
- MongoDB connection errors → Verify `MONGODB_URI` is set and the database/collection exist. Avoid embedding credentials directly in code.

---

## Development notes
- The app uses TF‑IDF to retrieve a context lyric similar to your prompt and then uses a next‑word model to generate text from that context.
- Temperature controls creativity: lower = safer/predictable, higher = more diverse.

---

## License
No license declared yet. If you intend to open‑source this project, add a `LICENSE` file (e.g., MIT) at the repository root.
