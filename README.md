# pdf_chatbot_llama
A Streamlit-based chatbot that allows users to upload PDF files and ask questions about their content. The application utilizes Sentence Transformers for semantic search and Llama3.2 via Ollama for generating responses. Note: Ollama must be running in the background.
# PDF Chatbot with Llama3.2

This is a Streamlit-based chatbot that allows users to upload PDF files and ask questions about their content. The application extracts text from the uploaded PDF, performs semantic search using Sentence Transformers, and generates responses using Llama3.2 via Ollama.

## Features
- Upload a PDF file and extract text from it.
- Use Sentence Transformers (`all-MiniLM-L6-v2`) to generate embeddings for semantic search.
- Retrieve relevant text chunks from the PDF to answer user queries.
- Utilize Llama3.2 via Ollama for AI-generated responses.
- Runs as a Streamlit web application.

## Prerequisites
- **Ollama must be running in the background** before using the chatbot.
- Install the required dependencies using `pip install -r requirements.txt`.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/pdf-chatbot.git
   cd pdf-chatbot


2.Install the required Python packages:   
pip install -r requirements.txt


3.Ensure that Ollama is installed and running
ollama serve

