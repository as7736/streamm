import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
import spacy
from rapidfuzz import fuzz
import os
import spacy



nlp = spacy.load("en_core_web_sm")



# ===================== Load Models & Dataset =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
model.eval()

# Load your CSV files
df1 = pd.read_csv("final amazon.csv")
df2 = pd.read_csv("combined_clean_preview.csv")

for col in df1.columns:
    if col not in df2.columns:
        df2[col] = ""
for col in df2.columns:
    if col not in df1.columns:
        df1[col] = ""

df = pd.concat([df1, df2], ignore_index=True)
df['combined'] = df[['title', 'description', 'categories', 'manufacturer',
                     'top_review', 'features', 'brand', 'combined_clean']].fillna('').agg('  '.join, axis=1)
df['combined_clean'] = df['combined'].str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True)
df['combined_clean'] = df['combined_clean'].str.replace(r'[\.\_]', ' ', regex=True)

catalog_vocab = set()
df['combined_clean'].dropna().apply(lambda x: catalog_vocab.update(x.split()))

def split_and_expand(text):
    return set(text.replace('.', ' ').split())

def generate_product_expansions(df):
    product_expansions = set()
    for _, row in df.iterrows():
        combined_text = f"{row['title']} {row['description']} {row['categories']} {row['manufacturer']} {row['features']} {row['brand']} {row['combined_clean']} "
        product_expansions.update(split_and_expand(combined_text))
    return product_expansions

product_expansions = generate_product_expansions(df)

# ===================== Correction Function =====================
def enhanced_correction_with_suggestions(text, product_expansions):
    tokens = text.lower().split()
    corrected_tokens = []
    for token in tokens:
        if not token:
            continue
        if token in product_expansions:
            corrected_tokens.append(token)
        else:
            suggestions = [term for term in product_expansions if fuzz.ratio(token, term) > 75]
            if suggestions:
                corrected_tokens.append(suggestions[0])
            else:
                corrected_tokens.append(token)
    return " ".join(corrected_tokens)

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Keyword Spell Correction", layout="centered")

st.markdown("## 🔍 Keywords")
st.markdown("Type your keyword below. Press Enter to get the corrected version.")

query = st.text_input("")

if query:
    corrected = enhanced_correction_with_suggestions(query, product_expansions)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ❌ Original")
        st.write(query)
    with col2:
        st.markdown("### ✅ Corrected")
        st.write(corrected)
