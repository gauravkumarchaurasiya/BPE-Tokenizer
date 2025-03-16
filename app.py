import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
import pickle
import os
from collections import defaultdict
from tokenizer import BPETokenizer
# Function to scrape Wikipedia (cached)
def scrape_wikipedia():
    cache_file = "wikipedia_text.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Quantum_computing",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Blockchain",
    ]
    text_data = ""
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text_data += " ".join([p.text for p in paragraphs[:5]]) + "\n"
    
    with open(cache_file, "wb") as f:
        pickle.dump(text_data, f)
    return text_data

# Load or train tokenizer (cached)
def load_or_train_tokenizer():
    tokenizer_file = "bpe_tokenizer.pkl"
    if os.path.exists(tokenizer_file):
        with open(tokenizer_file, "rb") as f:
            return pickle.load(f)
    
    text = scrape_wikipedia()
    bpe = BPETokenizer(vocab_size=1256)
    bpe.train(text)
    
    with open(tokenizer_file, "wb") as f:
        pickle.dump(bpe, f)
    return bpe

# Load pre-trained tokenizer
bpe = load_or_train_tokenizer()

# Streamlit UI

st.set_page_config(page_title="BPE Tokenizer | by Gaurav", 
                   page_icon="📖", 
                   layout="centered")

# Add SEO tags
st.markdown("""
    <meta name="description" content="BPE Tokenizer trained on Wikipedia. Tokenize and decode text easily!">
    <meta name="keywords" content="BPE Tokenizer,LLM,Chatgpt tokenizer,tiktoken, NLP, AI, Tokenization, Streamlit, Text Processing">
    <meta name="author" content="Gaurav Kumar Chaurasiya">
    <meta property="og:title" content="BPE Tokenizer">
    <meta property="og:description" content="Interactive BPE Tokenizer trained on Wikipedia. Encode & Decode text easily!">
    <meta property="og:type" content="website">
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>BPE Tokenizer</h1>", unsafe_allow_html=True)

st.write("A Byte Pair Encoding (BPE) tokenizer trained on 10 Wikipedia articles with vocab size 1256 [1000 merges]")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔠 Encode Text")
    user_text = st.text_area("Enter text:", "Hello World",height=170)
    if st.button("Tokenize", key="tokenize_btn"):
        tokens = bpe.encode(user_text)
        token_count = len(tokens.split()) 
        st.write(f"Token Count: {token_count}")
        st.text_area("Tokenized Output:", tokens, height=100)

with col2:
    st.subheader("📝 Decode Tokens")
    user_tokens = st.text_area("Enter tokens (space-separated):",height=170)
    if st.button("Decode", key="decode_btn"):
        decoded_text = bpe.decode(user_tokens)
        st.write(f"Token Count: {len(decoded_text)}")
        st.text_area("Decoded Output:", decoded_text, height=100)

with st.expander("ℹ️ View Tokenizer Details"):
    show_vocab = st.checkbox("Show Vocabulary")
    show_merges = st.checkbox("Show Merges")
    
    if show_vocab:
        st.write("### Vocabulary")
        formatted_vocab = ""
        for idx, (k, v) in enumerate(bpe.vocab.items()):
            if idx < 32:  # First 31 entries
                hex_value = ' '.join([f"0x{byte:02x}" for byte in v])  # This will convert bytes to '0x80', '0x81', etc.
                formatted_vocab += f"{k}: {hex_value}\n"
            else: 
                decoded_value = v.decode('utf-8', 'ignore')  
                formatted_vocab += f"{k}: {decoded_value}\n"

        st.text_area("Vocab", formatted_vocab, height=200)
    if show_merges:
        st.write("### Merges")
        formatted_merges = "\n".join([f"{str(k)}: {v}" for k, v in bpe.merges.items()])
        st.text_area("Merges", formatted_merges, height=200)