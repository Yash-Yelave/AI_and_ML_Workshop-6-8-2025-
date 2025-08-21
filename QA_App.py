import streamlit as st
from llama_cpp import Llama
import os
import PyPDF2

MODEL_PATH = "Llama-2-7b-chat-hf-GGUF-Q4_K_M.gguf"

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Please place the model in the app directory.")
        return None
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,             #Context window size
            n_threads=4,            #Uses 4 CPU threads.
            n_batch=512,            #Batch size for processing
            n_gpu_layers=0,         #Number of GPU layers (0 for CPU only)
        )
        return llm
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
    
# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to generate answer using LLaMA model
def ask_pdf_question(llm, context, question):
    prompt = f"""### Context:\n{context}\n\n### Human: {question}\n### Assistant:"""
    try:
        response = llm(prompt,
                      max_tokens=200,                               # Maximum tokens in the response
                      temperature=0.7,                              # Controls randomness in the output (higher = more creative).
                      stop=["### Human:", "### Assistant:"])        # Stop sequences to end the response
        answer = response['choices'][0]['text'].strip()
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

st.set_page_config(page_title="PDF QA System Using LLM", layout="centered")
st.title("PDF Question-Answer System (Llama LLM)")
st.info("Wait While Your Model is Loading...")

llm = load_model()
pdf_text = ""

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("PDF loaded successfully!")
    st.text_area("Extracted PDF Text (editable, optional)", value=pdf_text, height=200)

if llm is not None and pdf_text:
    question = st.text_area("Enter your question:", height=80)
    if st.button("Get Answer") and question.strip():
        with st.spinner("Generating answer..."):
            answer = ask_pdf_question(llm, pdf_text, question)
            st.markdown(f"**Answer:**\n{answer}")
else:
    if llm is None:
        st.warning("The Llama model could not be loaded. Please check the model file.")
    elif not pdf_text:
        st.info("Please upload a PDF file to begin.")
