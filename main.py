import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import re
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_artifacts():
    """Load the model, tokenizer, and vectorizer once."""
    try:
        # 'compile=False' is used because we are not training, only doing inference.
        model = load_model('models/rag_lyrics_model.h5', compile=False)
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        # The max_sequence_length should match the one used during training in the notebook.
        max_sequence_length = 100 
        
        return model, tokenizer, tfidf_vectorizer, max_sequence_length
    except (FileNotFoundError, AttributeError, KeyError) as e:
        st.error(f"Error loading artifacts: {e}. Please ensure all model files ('rag_lyrics_model.h5', 'tokenizer.pickle', 'tfidf_vectorizer.pkl') are in the 'models' directory.")
        return None, None, None, None

model, tokenizer, tfidf_vectorizer, max_sequence_length = load_artifacts()

# Load Lyric Data for Retrieval 
@st.cache_data
def load_data():
    """Load and preprocess the lyrics dataset from both CSV and MongoDB."""
    try:
        # Connect to MongoDB Atlas
        # Replace with your connection string if it differs.
        client = MongoClient("mongodb+srv://mayankkr0311_db_user:Lp4b3Jp5SGzaUBOu@cluster0.p1pyttx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        db = client["food"]
        lyrics_collection = db["lyrics"]
        
        # Load data from both sources 
        mongo_df = pd.DataFrame(list(lyrics_collection.find()))
        csv_df = pd.read_csv('ArianaGrande.csv')
        
        # Combine, clean, and preprocess 
        combined_df = pd.concat([csv_df, mongo_df], ignore_index=True)
        # Ensure 'Lyric' column exists and handle potential missing values
        if 'Lyric' not in combined_df.columns:
            st.error("The loaded data does not contain a 'Lyric' column.")
            return pd.DataFrame()
            
        combined_df['Lyric'] = combined_df['Lyric'].fillna('').str.lower()
        
        # The hyphen '-' is moved to the end of the character set to be treated literally.
        combined_df['Lyric'] = combined_df['Lyric'].apply(lambda x: re.sub(r"[^a-z0-9\s.,!?;:'\"-]", '', x))
        
        st.success(f"Lyrics data loaded successfully. Total rows: {combined_df.shape[0]}")
        return combined_df
    except FileNotFoundError:
        st.error("Error: `ArianaGrande.csv` not found. Please place it in the same directory as the app.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error connecting to MongoDB or processing data: {e}")
        return pd.DataFrame()

df = load_data()

# Create TF-IDF matrix for retrieval only if data and vectorizer are loaded
if tfidf_vectorizer is not None and not df.empty:
    tfidf_matrix = tfidf_vectorizer.transform(df['Lyric'])
else:
    tfidf_matrix = None

# RAG Functions 
def retrieve_lyric(prompt, vectorizer, matrix, dataframe):
    """Retrieves the most relevant lyric from the dataset based on a prompt."""
    preprocessed_prompt = prompt.lower()
    prompt_vector = vectorizer.transform([preprocessed_prompt])
    cosine_similarities = cosine_similarity(prompt_vector, matrix).flatten()
    most_similar_index = cosine_similarities.argmax()
    return dataframe['Lyric'].iloc[most_similar_index]

def generate_lyrics(model, tokenizer, seed_text, max_len, num_words=100, temperature=0.8):
    """Generate new lyrics using the same logic as in the notebook."""
    generated = []
    effective_max_len = max_len - 1

    for _ in range(num_words):
        # Convert current seed text to sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad to the sequence length used in training
        token_list = pad_sequences([token_list], maxlen=effective_max_len, padding='pre')

        # Model output shape: (1, seq_len, vocab_size) -> take last timestep probabilities
        preds = model.predict(token_list, verbose=0)[0, -1, :]

        # Normalize probabilities to sum to 1
        preds = preds / (np.sum(preds) + 1e-9)

        # Apply temperature sampling (same as notebook)
        preds = np.log(preds + 1e-9) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        # Sample next word index from the distribution
        next_index = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word.get(next_index, "")

        # Skip unknown / empty tokens
        if not next_word:
            continue

        seed_text += " " + next_word
        generated.append(next_word)

    return " ".join(generated)

# Streamlit User Interface 
st.title("🎵 AI Lyrics Generator with RAG")

# Main app logic runs only if all components are successfully loaded
if all(v is not None for v in [model, tokenizer, tfidf_vectorizer, max_sequence_length]) and not df.empty and tfidf_matrix is not None:
    input_prompt = st.text_input("Enter your prompt:", "I'm feeling happy today")

    ## UI controls for generation parameters
    temperature = st.slider(
        "Creativity:",
        min_value=0.1, max_value=1.5, value=0.3, step=0.1,
        help="Lower values are more predictable; higher values are more creative."
    )

    num_words_to_generate = st.slider(
        "Number of words to generate:",
        min_value=150, max_value=600, value=100, step=10
    )

    if st.button("Generate Lyrics"):
        if input_prompt.strip():
            with st.spinner("Retrieving context and generating lyrics..."):
                # Retrieve the most relevant lyric to use as context
                retrieved_context = retrieve_lyric(input_prompt, tfidf_vectorizer, tfidf_matrix, df)
                st.info(f"**Retrieved Context:** *{retrieved_context[:200]}...*")

                # Generate new lyrics based on the retrieved context
                generated_text = generate_lyrics(model, tokenizer, retrieved_context, max_sequence_length, num_words_to_generate, temperature)
                
                st.subheader("Generated Lyrics:")
                st.write(generated_text if generated_text else "Could not generate lyrics. Please try a different prompt or adjust the temperature.")
        else:
            st.warning("Please enter a prompt to generate lyrics.")
else:
    st.error("Application could not be initialized. Please check the file paths and error messages above.")

