import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import nltk

# Download sentence tokenizer if not already present
nltk.download('punkt', quiet=True)

# --- Streamlit UI ---
st.set_page_config(page_title="📘 Sentence-based PDF Q/A Chatbot")
st.title("📘 Gemini PDF Q/A Chatbot using Sentence-based Chunking")

# --- Gemini API Key ---
api_key = st.text_input("🔑 Enter your Gemini API key:", type="password")

if api_key:
    genai.configure(api_key=api_key)

    # --- Upload PDF ---
    uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])
    if uploaded_file:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

        # --- Sentence-based Chunking ---
        sentences = nltk.sent_tokenize(text)
        chunk_size = 5  # number of sentences per chunk
        chunks = [
            " ".join(sentences[i:i+chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]
        st.success(f"✅ Document loaded with {len(chunks)} sentence chunks.")

        # --- User Question ---
        question = st.text_input("💬 Ask a question about the document:")

        if question:
            # Simple matching: select the chunk with most overlapping words
            best_chunk = ""
            max_score = 0
            for chunk in chunks:
                score = sum(word.lower() in chunk.lower() for word in question.split())
                if score > max_score:
                    max_score = score
                    best_chunk = chunk

            if best_chunk:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"Answer the question based only on this text:\n\n{best_chunk}\n\nQuestion: {question}"
                response = model.generate_content(prompt)

                st.markdown("### 🤖 Answer:")
                st.write(response.text)
            else:
                st.warning("No relevant chunk found.")
    else:
        st.info("Please upload a PDF file.")
else:
    st.info("Enter your Gemini API key to start.")
