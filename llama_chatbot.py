import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
import subprocess
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import tempfile
import os
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    # Force UTF-8 encoding
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class PDFChatbot:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentences = []
        self.embeddings = None

    def clean_text(self, text):
        """Clean text by removing problematic characters."""
        text = text.replace('\u2212', '-')  # Replace minus sign
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        return text

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

            reader = PdfReader(tmp_file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            os.unlink(tmp_file_path)
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def create_embeddings(self, text):
        """Create embeddings from text."""
        try:
            self.sentences = [self.clean_text(sent.strip()) 
                              for sent in text.split('.') if sent.strip()]
            self.embeddings = self.embedder.encode(self.sentences)
            return True
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return False

    def get_relevant_context(self, question, max_chunks=3):
        """Get most relevant text chunks for the question."""
        try:
            question = self.clean_text(question)
            question_embedding = self.embedder.encode([question])[0]
            similarities = cosine_similarity([question_embedding], self.embeddings)[0]
            top_indices = np.argsort(similarities)[-max_chunks:]
            relevant_text = ' '.join([self.sentences[i] for i in top_indices])
            return relevant_text
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return ""

    def llama_chat(self, prompt, max_length=2000):
        """Interface with Llama3.2 via Ollama."""
        try:
            # Clean and truncate prompt
            cleaned_prompt = self.clean_text(prompt[:max_length])

            # Set up process with explicit encoding
            startupinfo = None
            if sys.platform.startswith('win'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            # Run Llama3.2 through Ollama with proper encoding
            process = subprocess.Popen(
                ['ollama', 'run', 'llama3.2'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                startupinfo=startupinfo
            )

            # Communicate with the process
            stdout, stderr = process.communicate(input=cleaned_prompt)

            if process.returncode != 0:
                logger.error(f"Ollama process error: {stderr}")
                return f"Error: Failed to generate response. Please check if Ollama is running. Error: {stderr}"

            if not stdout.strip():
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

            return stdout.strip()

        except Exception as e:
            logger.error(f"Unexpected error in llama_chat: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

def main():
    st.set_page_config(page_title="PDF Chatbot with Llama3.2", layout="wide")
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("📚 PDF Chatbot with Llama3.2")
    st.markdown("""
    Upload a PDF file and ask questions about its content.
    The chatbot will use AI to provide relevant answers based on the document.
    """)

    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        This chatbot uses:
        - Llama3.2 for generating responses
        - Sentence Transformers for semantic search
        - PyPDF2 for PDF processing
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.pdf_processed = False

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        try:
            if not st.session_state.pdf_processed:
                with st.spinner("Processing PDF file..."):
                    pdf_text = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)
                    if st.session_state.chatbot.create_embeddings(pdf_text):
                        st.session_state.pdf_processed = True
                        st.success("PDF processed successfully!")
                    else:
                        st.error("Failed to process PDF.")
                        return

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return

        st.header("💬 Chat")
        
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
        
        user_input = st.text_input("Ask a question about the PDF:", key="user_input")
        
        if user_input:
            with st.spinner("Thinking..."):
                relevant_context = st.session_state.chatbot.get_relevant_context(user_input)
                
                prompt = f"""Context from the PDF:
                {relevant_context}

                Question: {user_input}

                Please provide a response based on the context above."""
                
                response = st.session_state.chatbot.llama_chat(prompt)
                
                st.session_state.chat_history.append((user_input, response))
                
                st.markdown(f"**You:** {user_input}")
                st.markdown(f"**Assistant:** {response}")

        with st.expander("Show PDF Content"):
            if hasattr(st.session_state.chatbot, 'sentences'):
                st.write("\n".join(st.session_state.chatbot.sentences))

if __name__ == "__main__":
    main()
