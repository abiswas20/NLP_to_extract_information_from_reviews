import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache
def bert_sent_embed(data):
    model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    embeddings = model.encode(data, show_progress_bar=True)
    return embeddings
