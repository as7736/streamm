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

# ===================== Soundex Function =====================
def soundex(word):
    word = word.lower()
    codes = {
        'a': '', 'e': '', 'i': '', 'o': '', 'u': '', 'y': '', 'h': '', 'w': '',
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    if not word:
        return ""
    first_letter = word[0].upper()
    encoded = first_letter
    for char in word[1:]:
        encoded += codes.get(char, '')
    encoded = encoded.replace('0', '')
    result = encoded[0]
    for char in encoded[1:]:
        if char != result[-1]:
            result += char
    return (result + '000')[:4]

# ===================== Correction function (normal) =====================
def correct_query(text, vocab, token_freq):
    tokens = text.lower().split()
    corrected_tokens = []
    uncorrected_tokens = []

    for token in tokens:
        if not token:
            continue
        if token in vocab:
            corrected_tokens.append(token)
        else:
            matches = []
            for word in vocab:
                score = fuzz.ratio(token, word)
                if score > 74:
                    matches.append((word, score))
            if matches:
                best = sorted(matches, key=lambda x: (-token_freq[x[0]], -x[1]))[0][0]
                corrected_tokens.append(best)
            else:
                corrected_tokens.append(token)  # keep original
                uncorrected_tokens.append(token)

    corrected_text = " ".join(corrected_tokens)
    return corrected_text, uncorrected_tokens

# ===================== Soundex + Meaning based Suggestion =====================
def get_soundex_suggestions_whole_phrase(query_text, vocab):
    if not query_text.strip():
        return []

    query_soundex = soundex(query_text.replace(' ', ''))
    candidates = []

    for word in vocab:
        if not word:
            continue
        word_soundex = soundex(word)
        if query_soundex == word_soundex:
            meaning_score = fuzz.partial_ratio(query_text, word)
            candidates.append((word, meaning_score))

    if not candidates:
        return []

    # Rank better: First meaning score, then freq
    candidates_sorted = sorted(candidates, key=lambda x: (-x[1], x[0]))

    # Limit top 5 best suggestions
    top_suggestions = [word for word, score in candidates_sorted[:5]]

    return top_suggestions

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Spell Correction", layout="centered")
st.title("🔍 E-commerce Search Spell Corrector")

# Load vocab
with st.spinner("Loading and preparing data..."):
    vocab, token_freq = load_data()

# Input box
query = st.text_input("Enter your product search query:")

if query:
    corrected_text, uncorrected_tokens = correct_query(query, vocab, token_freq)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔡 Original")
        st.code(query, language="text")
    with col2:
        st.markdown("### ✅ Corrected")
        st.code(corrected_text, language="text")

    # Combined query for better Soundex
    combined_query = corrected_text.strip()

    soundex_suggestions = get_soundex_suggestions_whole_phrase(combined_query, vocab)

    if soundex_suggestions:
        st.markdown("### ✨ Additional Suggestions (Soundex + Meaning Ranking):")
        st.write(f"**{combined_query}** → {', '.join(soundex_suggestions)}")
