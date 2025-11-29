import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Load Pre-trained Models and Artifacts ---
@st.cache_resource
def load_artifacts():
    """Load the model, tokenizer, and vectorizer once."""
    try:
        model = load_model('models/rag_lyrics_model.h5', compile=False)
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        # Get max_sequence_length from the model's input shape if possible
        max_sequence_length = model.input_shape[1]
        return model, tokenizer, tfidf_vectorizer, max_sequence_length
    except (FileNotFoundError, AttributeError, KeyError) as e:
        st.error(f"Error loading artifacts: {e}. Please ensure all model files are in the 'models' directory and are valid.")
        return None, None, None, None


model, tokenizer, tfidf_vectorizer, max_sequence_length = load_artifacts()


# --- Load Lyric Data for Retrieval ---
@st.cache_data
def load_data():
    """Load the lyrics dataset."""
    try:
        df = pd.read_csv('ArianaGrande.csv')
        df['Lyric'] = df['Lyric'].fillna('')
        return df
    except FileNotFoundError:
        st.error("Error: `ArianaGrande.csv` not found. Please place it in the same directory as the app.")
        return pd.DataFrame()


df = load_data()


if all(v is not None for v in [model, tokenizer, tfidf_vectorizer, max_sequence_length]) and not df.empty:
    tfidf_matrix = tfidf_vectorizer.transform(df['Lyric'])

    # --- RAG Functions ---
    def retrieve_lyric(prompt, vectorizer, matrix, dataframe):
        """Retrieves the most relevant lyric based on a prompt."""
        preprocessed_prompt = prompt.lower()
        prompt_vector = vectorizer.transform([preprocessed_prompt])
        cosine_similarities = cosine_similarity(prompt_vector, matrix).flatten()
        most_similar_index = cosine_similarities.argmax()
        return dataframe['Lyric'].iloc[most_similar_index]

    def generate_lyrics(model, tokenizer, retrieved_lyric, max_len, num_words_to_generate=50, temperature=0.8):
        """Generates lyrics with temperature sampling, handling unknown words gracefully."""
        seed_text = retrieved_lyric
        generated_lyrics = []
        unknown_word_counter = 0

        for _ in range(num_words_to_generate):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_len, padding='pre', truncating='pre')
            
            predicted_probabilities = model.predict(token_list, verbose=0)[0]
            
            # This assumes the model output is for the full sequence.
            # Get probabilities for the last token.
            last_token_probabilities = predicted_probabilities[-1]

            # --- TEMPERATURE SAMPLING LOGIC ---
            last_token_probabilities = np.log(last_token_probabilities + 1e-9) / temperature
            exp_preds = np.exp(last_token_probabilities)
            last_token_probabilities = exp_preds / np.sum(exp_preds)
            
            predicted_word_index = np.random.choice(len(last_token_probabilities), p=last_token_probabilities)
            
            output_word = tokenizer.index_word.get(predicted_word_index, "")
            
            # --- MODIFIED LOGIC TO HANDLE UNKNOWN WORDS ---
            if not output_word:
                unknown_word_counter += 1
                # If the model gets stuck, break to avoid an infinite loop
                if unknown_word_counter > 5:
                    st.warning("Model generated multiple unknown words, stopping generation to avoid errors.")
                    break
                continue  # Skip this iteration and try generating the next word
            
            unknown_word_counter = 0  # Reset counter on a successful word
            seed_text += " " + output_word
            generated_lyrics.append(output_word)

        return " ".join(generated_lyrics)

    # --- Streamlit UI ---
    st.title("🎵 Generative AI Lyrics Generator")
    st.write("Enter a theme or a line, and the RAG model will generate lyrics for you.")

    input_prompt = st.text_input("Enter your prompt:", "I'm feeling sad")

    temperature = st.slider(
        "Creativity (Temperature):",
        min_value=0.1,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="Lower values are more predictable; higher values are more creative."
    )

    num_words_to_generate = st.slider(
        "Number of words to generate:",
        min_value=20,
        max_value=5000,
        value=100,
        step=10
    )

    if st.button("Generate Lyrics"):
        if input_prompt.strip():
            with st.spinner("Generating..."):
                retrieved = retrieve_lyric(input_prompt, tfidf_vectorizer, tfidf_matrix, df)
                st.info(f"**Retrieved Context:** *{retrieved[:150]}...*")
                
                # Pass the dynamically loaded max_sequence_length
                generated = generate_lyrics(model, tokenizer, retrieved, max_sequence_length, num_words_to_generate, temperature)
                
                st.subheader("Generated Lyrics:")
                st.write(generated if generated else "Could not generate lyrics. The model may need retraining or the context was too short.")
        else:
            st.warning("Please enter a prompt.")
else:
    st.error("Could not initialize the app. Please check the console for file loading errors.")
