import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np


@st.cache_resource
def load_embedding_model():
    # Loads only once per session
    return SentenceTransformer("all-MiniLM-L6-v2")


class EmbeddingModel:
    def __init__(self):
        self.model = load_embedding_model()

    @st.cache_data(show_spinner=False)
    def encode(self, texts_tuple):
        """
        texts_tuple must be tuple (hashable for caching)
        """
        texts = list(texts_tuple)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return np.array(embeddings)