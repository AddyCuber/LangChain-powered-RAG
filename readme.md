Perfect — here’s your README.md fully cleaned up, no emojis, proper markdown formatting, and neat indentation. You can paste this directly:

⸻


# Gemini PDF Chatbot

This is a LangChain-powered PDF chatbot that uses Gemini models for question answering over document chunks. FAISS is used for vector storage, and the entire pipeline is built using LangChain’s RAG components.

---

## Features

- Upload and process multiple PDF files  
- Chunk large text into manageable pieces  
- Generate embeddings using `text-embedding-004`  
- Store and search embeddings with FAISS  
- Answer questions using Gemini 1.5 Flash with context-aware responses  

---

## Tech Stack

- Frontend: Streamlit  
- Vector DB: FAISS  
- LLM: Gemini via `langchain_google_genai`  
- Embeddings: `GoogleGenerativeAIEmbeddings`  
- PDF Processing: `PyPDF2`  
- Environment: Python 3.10+  

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/AddyCuber/LangChain-powered-RAG.git
cd LangChain-powered-RAG

2. Set up a virtual environment

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

Required packages:
	•	streamlit
	•	PyPDF2
	•	python-dotenv
	•	langchain
	•	faiss-cpu
	•	chromadb
	•	langchain_google_genai
	•	langchain-community
	•	google-generativeai

4. Add your API Key

Create a .env file in the project root:

GOOGLE_API_KEY=your_api_key_here

5. Run the app

Make sure you are in the correct directory:

streamlit run app.py


⸻

Project Structure

.
├── app.py                  # Main Streamlit app
├── .env                    # Contains Google API key (should be ignored by Git)
├── faiss_index/            # Stores FAISS vector index
├── requirements.txt        # All dependencies
└── README.md               # Project documentation


⸻

How It Works
	•	Extracts text from uploaded PDFs using PyPDF2
	•	Splits text using LangChain’s RecursiveCharacterTextSplitter
	•	Embeds chunks using GoogleGenerativeAIEmbeddings
	•	Stores and retrieves vectors using FAISS
	•	Responds to user queries using ChatGoogleGenerativeAI with context from PDF content

⸻

Challenges & Solutions
	•	Problem: Streamlit could not find app.py
Solution: Ensured the correct directory using cd gemini-project before running the app
	•	Problem: ModuleNotFoundError: No module named 'pypdf2'
Solution: Installed correct package with pip install PyPDF2
	•	Problem: ModuleNotFoundError: No module named 'langchain_google_genai'
Solution: Installed it with pip install langchain_google_genai
	•	Problem: ModuleNotFoundError: No module named 'google.generativeai'
Solution: Installed with pip install google-generativeai
	•	Problem: ModuleNotFoundError: No module named 'langchain_community'
Solution: Installed langchain-community and updated imports
	•	Problem: ValidationError: model Field required
Solution: Replaced model_name= with model= in class initializations
	•	Problem: ValueError: Pickle deserialization warning
Solution: Added allow_dangerous_deserialization=True to FAISS.load_local()
	•	Problem: ImportError: Could not import faiss python package
Solution: Installed with pip install faiss-cpu
	•	Problem: RuntimeError: could not open faiss_index/index.faiss for reading
Solution: Made sure a PDF was uploaded and processed before querying
	•	Problem: TypeError: FAISS.save_local() got unexpected keyword
Solution: Removed unsupported keyword argument — allow_dangerous_deserialization is only for load_local()
	•	Problem: 'GoogleGenerativeAI' object has no attribute 'embed_documents'
Solution: Used correct class: GoogleGenerativeAIEmbeddings
	•	Problem: 404 models/gemini-pro is not found
Solution: Switched to gemini-1.5-flash, a supported model
	•	Problem: IndexError: list index out of range
Solution: Added checks for empty PDFs and chunks with Streamlit warnings
	•	Problem: Environment/path mismatch
Solution: Emphasized correct directory and virtual environment usage

⸻

Notes
	•	Only works for PDFs containing extractable text (not scanned images)
	•	Make sure your Google API key has access to Gemini models
	•	The FAISS vector index must be created before asking questions

⸻

Future Improvements
	•	Add OCR support for scanned PDFs
	•	Enable multi-turn chat memory
	•	Polish the UI and add a document viewer
	•	Dockerize for containerized deployment
