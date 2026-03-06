import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# =============================
# Load Files
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "incident_encoder_model.h5")
embeddings_path = os.path.join(BASE_DIR, "incident_embeddings.pickle")
maxlen_path = os.path.join(BASE_DIR, "max_len.pickle")
resolution_path = os.path.join(BASE_DIR, "resolution_data.pickle")
training_path = os.path.join(BASE_DIR, "training_data.pickle")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pickle")


# Load model
encoder_model = load_model(model_path, compile=False)

# Load data
with open(embeddings_path, "rb") as f:
    incident_embeddings = pickle.load(f)

with open(maxlen_path, "rb") as f:
    max_len = pickle.load(f)

with open(resolution_path, "rb") as f:
    y_train = pickle.load(f)

with open(training_path, "rb") as f:
    X_train = pickle.load(f)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)


# =============================
# Streamlit UI
# =============================

st.title("🛠 Incident Resolution Recommender System")

st.write("Enter an incident description to get recommended resolutions.")

query = st.text_area("Incident Description")


# =============================
# Recommendation Function
# =============================

def recommend_resolution(query, top_k=3):

    seq = tokenizer.texts_to_sequences([query])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    query_embedding = encoder_model.predict(padded)

    # Normalize embeddings
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    incident_embeddings_norm = incident_embeddings / np.linalg.norm(
        incident_embeddings, axis=1, keepdims=True
    )

    similarity = np.dot(incident_embeddings_norm, query_embedding.T).flatten()

    top_indices = similarity.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:

        if idx < len(X_train):

            incident_text = tokenizer.sequences_to_texts([X_train[idx]])[0]
            incident_text = incident_text.replace("<OOV>", "")

            resolution_text = y_train.iloc[idx]["resolution"]

            results.append({
                "Incident": incident_text,
                "Resolution": resolution_text,
                "Similarity Score": float(similarity[idx])
            })

    return pd.DataFrame(results)


# =============================
# Button Action
# =============================

if st.button("Get Resolution"):

    if query.strip() == "":
        st.warning("Please enter an incident description")

    else:

        with st.spinner("Finding similar incidents..."):

            results = recommend_resolution(query)

        st.subheader("Top Recommended Resolutions")

        for i, row in results.iterrows():

            st.markdown(f"### Recommendation {i+1}")

            st.markdown("**Similar Incident:**")
            st.info(row["Incident"])

            st.markdown("**Resolution:**")
            st.success(row["Resolution"])

            st.caption(f"Similarity Score: {round(row['Similarity Score'],3)}")

            st.divider()
