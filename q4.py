import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Step 1: Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def sentence_chunker(sentences, chunk_size):
    output = []
    cur_chunk = []

    for sent in sentences:
        cur_chunk.append(sent)
        if len(cur_chunk) == chunk_size:
            output.append(" ".join(cur_chunk))
            cur_chunk = []

    if cur_chunk:
        output.append(" ".join(cur_chunk))

    return output

if uploaded_file is not None:
    # Step 2: Extract text from PDF
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "

    st.success("PDF text extracted successfully.")

    # Step 3: Sentence tokenization
    sentences = sent_tokenize(full_text)

    st.subheader("Sample Extracted Sentences (Index 58 to 68)")

    if len(sentences) >= 69:
        sample_sentences = sentences[58:69]

        for i, s in enumerate(sample_sentences, start=58):
            st.write(f"**Sentence {i}:** {s}")
    else:
        st.warning("The PDF does not contain enough sentences (minimum 69 required).")
        sample_sentences = []

    # Step 4: Semantic sentence chunking
    if sample_sentences:
        st.subheader("Semantic Sentence Chunks")

        chunk_size = st.number_input(
            "Number of sentences per chunk",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        if st.button("Create sentence chunks"):
            chunks = sentence_chunker(sample_sentences, int(chunk_size))
            st.success(f"Number of sentence chunks = {len(chunks)}")

            for idx, chunk in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {idx}**")
                st.write(chunk)
                st.markdown("---")
else:
    st.info("Please upload a PDF file to begin.")

