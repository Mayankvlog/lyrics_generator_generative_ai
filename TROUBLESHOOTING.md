# Streamlit App Connection Troubleshooting Guide

## Issue: Cannot connect to http://127.0.0.1:8501

### **Solution 1: Run Streamlit Directly (Windows)**

```bash
# Navigate to project directory
cd c:\Users\mayan\Downloads\Addidas\Credit-card-fraud-detection-Data-science-project\lyrics_generator

# Activate virtual environment
myenv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run main.py
```

### **Solution 2: Using Docker (Requires Docker Desktop)**

#### Step 1: Install & Start Docker Desktop
1. Download from: https://www.docker.com/products/docker-desktop
2. Install and restart your computer
3. Open Docker Desktop application
4. Wait for Docker daemon to start (check system tray icon)

#### Step 2: Build and Run Container
```bash
cd c:\Users\mayan\Downloads\Addidas\Credit-card-fraud-detection-Data-science-project\lyrics_generator

# Build Docker image
docker build -t lyrics-generator:latest .

# Run container
docker run -p 8501:8501 lyrics-generator:latest
```

#### Step 3: Using Docker Compose
```bash
# Pull latest image from Docker Hub
docker-compose pull

# Start services
docker-compose up

# View logs
docker-compose logs -f app
```

### **Solution 3: Check Requirements & Dependencies**

```bash
# Activate virtual environment
myenv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt

# Verify Streamlit installation
streamlit --version
```

### **Solution 4: Verify Model Files Exist**

Check if these files are in the `models/` directory:
- ✅ `rag_lyrics_model.h5`
- ✅ `best_lyrics_model.h5`
- ✅ `tokenizer.pickle`
- ✅ `tfidf_vectorizer.pkl`

If missing, download or retrain models.

### **Solution 5: Fix MongoDB Connection**

In `main.py`, update MongoDB URI:
```python
# Option 1: Use local MongoDB
client = MongoClient("mongodb://localhost:27017")

# Option 2: Use your MongoDB Atlas connection string
client = MongoClient("YOUR_MONGO_URI_HERE")
```

### **Solution 6: Clear Streamlit Cache & Restart**

```bash
# Kill any running Streamlit processes
taskkill /IM streamlit.exe /F

# Clear Streamlit cache
rmdir /s /q %USERPROFILE%\.streamlit

# Restart app
streamlit run main.py
```

### **Common Errors & Fixes**

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'streamlit'` | Run: `pip install streamlit` |
| `FileNotFoundError: models/rag_lyrics_model.h5` | Ensure all model files exist in `models/` directory |
| `MongoDB connection refused` | Check MongoDB service is running or update connection string |
| `Port 8501 already in use` | Run: `netstat -ano \| findstr :8501` to find process, then `taskkill /PID <PID> /F` |
| `Docker daemon not running` | Open Docker Desktop application |

### **Verify Everything Works**

Once app starts, you should see:
```
You can now view your Streamlit app in your browser.

URL: http://localhost:8501
```

Then open http://127.0.0.1:8501 or http://localhost:8501 in your browser.

