import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# ==============================
# 1. Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "incident_encoder_model.h5")
embeddings_path = os.path.join(BASE_DIR, "incident_embeddings.pickle")
maxlen_path = os.path.join(BASE_DIR, "max_len.pickle")
resolution_path = os.path.join(BASE_DIR, "resolution_data.pickle")
training_path = os.path.join(BASE_DIR, "training_data.pickle")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pickle")


# ==============================
# 2. Load Model
# ==============================

encoder_model = load_model(model_path, compile=False)


# ==============================
# 3. Load Data
# ==============================

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


# ==============================
# 4. Recommendation Function
# ==============================

def recommend_resolution(query, top_k=3):

    seq = tokenizer.texts_to_sequences([query])

    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    query_embedding = encoder_model.predict(padded)

    similarity = cosine_similarity(query_embedding, incident_embeddings)[0]

    top_indices = similarity.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:

        if idx < len(X_train):

            # Convert sequence back to text
            incident_text = tokenizer.sequences_to_texts([X_train[idx]])[0]

            # Resolution handling
            if isinstance(y_train, pd.DataFrame):
                resolution_text = y_train.iloc[idx]["resolution"]
            else:
                resolution_text = y_train[idx]

            results.append({
                "Incident": incident_text,
                "Resolution": resolution_text,
                "Similarity Score": float(similarity[idx])
            })

    return pd.DataFrame(results)


# ==============================
# 5. Streamlit UI
# ==============================

st.title("🛠 Incident Resolution Recommender System")

st.write("Enter an incident description to get recommended resolutions.")

query = st.text_input("Incident Description")

submit = st.button("Get Resolution")


# ==============================
# 6. Run Recommendation
# ==============================

if submit:

    if query.strip() == "":
        st.warning("Please enter an incident description.")

    else:

        with st.spinner("Finding similar incidents..."):

            results = recommend_resolution(query)

        if len(results) == 0:

            st.error("No recommendations found.")

        else:

            st.success("Top Recommended Resolutions")

            for i, row in results.iterrows():

                st.markdown(f"### Recommendation {i+1}")

                st.write("**Similar Incident:**")
                st.write(row["Incident"])

                st.write("**Resolution:**")
                st.write(row["Resolution"])

                st.write("**Similarity Score:**", round(row["Similarity Score"], 3))

                st.divider()
