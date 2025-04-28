# app.py

import pandas as pd
from rapidfuzz import fuzz
from collections import Counter
import streamlit as st

# ===================== Load & Process Dataset =====================
@st.cache_data
def load_data():
    df1 = pd.read_csv("amazon-products.csv")
    df2 = pd.read_csv("electronics.csv")

    required_cols = ['title', 'categories', 'manufacturer', 'brand', 'category_code', 'brand01']
    for col in required_cols:
        if col not in df1.columns:
            df1[col] = ''
        if col not in df2.columns:
            df2[col] = ''

    df = pd.concat([df1, df2], ignore_index=True)
    df['combined'] = df[required_cols].fillna('').agg(' '.join, axis=1)
    df['combined_clean'] = df['combined'].str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True)

    all_tokens = " ".join(df['combined_clean'].dropna()).split()
    token_freq = Counter(all_tokens)
    vocab = set(token_freq.keys())

    return vocab, token_freq

# ===================== Spell Correction Function =====================
def enhanced_correction_with_freq(text, vocab, token_freq):
    tokens = text.lower().split()
    corrected_tokens = []

    for token in tokens:
        if not token:
            continue
        if token in vocab:
            corrected_tokens.append(token)
        else:
            matches = [(word, fuzz.ratio(token, word)) for word in vocab if fuzz.ratio(token, word) > 74]
            if matches:
                best = sorted(matches, key=lambda x: (-token_freq[x[0]], -x[1]))[0][0]
                corrected_tokens.append(best)
            else:
                corrected_tokens.append(token)
    return " ".join(corrected_tokens)

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Spell Correction", layout="centered")
st.title("🔍 E-commerce Search Spell Corrector")

# Load vocab
with st.spinner("Loading and preparing data..."):
    vocab, token_freq = load_data()

# Input box
query = st.text_input("Enter your product search query:")

# Show correction below
if query:
    corrected = enhanced_correction_with_freq(query, vocab, token_freq)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔡 Original ")
        st.code(query, language="text")
    with col2:
        st.markdown("### ✅ Corrected ")
        st.code(corrected, language="text")
