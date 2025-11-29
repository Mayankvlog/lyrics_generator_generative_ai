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
├── ArianaGrande.csv         # Sample dataset (must contain a `Lyric` column)
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

From inside the `lyrics_generator/` folder:

```bash
streamlit run main.py
```

This will open a local URL in your browser. Enter a prompt, pick temperature and number of words, and click “Generate Lyrics”.

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
